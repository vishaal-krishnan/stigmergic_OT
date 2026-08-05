"""
Adjoint Path Integral Control (APIC) - Algorithm 1 of the main text.

This file implements the iterative forward-backward optimization loop used
for every Snell-Descartes result in the paper.  Each `run_apic_loop` call
runs a sequence of stigmergic cycles, and each cycle has four stages:

    Stage 1 (forward pass)
        `simulate_forward_batch` integrates the reparametrized Langevin
        dynamics (Eq. 4) from source to target under the current pheromone
        field, applying the trail-following steering law (Eq. 2).

    Stage 2 (backward adjoint sweep)
        `integrate_costate` integrates the adjoint equations (Eq. 6)
        BACKWARD in arc length s along each forward trajectory, starting
        from the terminal conditions Gamma(1)=0, mu(1)=-grad Psi.

    Stage 3 (controlled backward pass)
        `simulate_controlled_backward_pass` returns from target to source
        under the feedback law (Eq. 7),  omega_ctrl(s) = -Gamma(s)/gamma,
        with gamma = beta * D_theta.

    Stage 4 (pheromone deposition)
        `downsample_recent_weighted_trajectories` builds the pheromone
        field consumed by the next cycle's forward pass, with a
        recency-weighted resampling of past controlled returns.

Physical parameters follow the paper:

    beta       Eq. 2 gain on the trail-following torque
    D_theta    angular diffusion coefficient
    eps_theta  time-scale separation (fast angular dynamics), eps_theta << 1
    gamma      Eq. 7 control gain, gamma = beta * D_theta (fluctuation-
               dissipation relation)

Notation.  The paper writes tildes over reparametrized variables; we drop
them in the code for readability, so `x`, `y`, `theta` mean the same as
X-tilde, Y-tilde, Theta-tilde in Eqs. 4-7 of the manuscript.

Sign convention on the costate.  The costate returned by
`integrate_costate` uses the paper's sign convention, i.e. Eq. 6 as
written and Eq. 7 in its literal form omega_ctrl(s) = - Gamma(s) / gamma.
An intermediate integration variable inside `integrate_costate` picks up
the opposite sign from a direct augmented-Lagrangian variation; the final
return is negated to align with the paper.  See the note in that function.
"""

import jax
import jax.numpy as jnp
from jax import grad, lax, random, vmap
from tqdm import trange


# =============================================================================
# Environment: refractive index nu(x, y)
# =============================================================================

def smooth_piecewise_nu(x, y, base=1.0, jump=10.0, steep=100.0, boundary=0.5):
    """Refractive index nu(x, y) with a smooth step across y = boundary.

    A sigmoid interpolates from `base` (below the boundary) to `jump` (above),
    with `steep` controlling the sharpness of the transition.  In the limit
    steep -> infinity this recovers the piecewise-constant medium used in the
    classical Snell-Descartes derivation.
    """
    s = 1.0 / (1.0 + jnp.exp(-steep * (y - boundary)))
    return base * (1.0 - s) + jump * s


# =============================================================================
# Pheromone field: kernel density estimate over deposited points
# =============================================================================

def compute_weighted_pheromone_gradient(points, weights, x, y, sigma):
    """Gradient of log phi(x, y), where phi is a weighted Gaussian KDE.

    The pheromone field phi is represented non-parametrically as a sum of
    isotropic Gaussians of width `sigma` centered at each deposited point.
    The small additive constant (0.01) inside the log is a numerical floor
    that keeps grad log phi finite where there are no nearby deposits.

    Returns a length-2 jnp.array [d/dx, d/dy] of log phi at (x, y).
    """
    def eval_log_phi(xx, yy):
        dx = xx - points[:, 0]
        dy = yy - points[:, 1]
        kernel = jnp.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
        return jnp.log(0.01 + jnp.sum(kernel * weights))

    gx = grad(lambda xx: eval_log_phi(xx, y))(x)
    gy = grad(lambda yy: eval_log_phi(x, yy))(y)
    return jnp.array([gx, gy])


# =============================================================================
# Trail-following potential E and its theta-derivatives (used in Eq. 6)
# =============================================================================
#
# Definition from the paper:
#     E(x, y, theta) = - beta * grad log phi(x, y) . (cos theta, sin theta)
#
# Its derivatives with respect to theta:
#     dE/dtheta       =  beta * ( d/dx log phi * sin theta
#                                 - d/dy log phi * cos theta )
#     d^2 E / dtheta^2 =  beta * ( d/dx log phi * cos theta
#                                  + d/dy log phi * sin theta )
#
# `partial_theta_E` is exactly Eq. 2 up to a minus sign: omega_tf = -dE/dtheta.

def _partial_theta_E(x, y, theta, pher_points, pher_weights, sigma_pher, beta):
    g = compute_weighted_pheromone_gradient(
        pher_points, pher_weights, x, y, sigma_pher
    )
    return beta * (g[0] * jnp.sin(theta) - g[1] * jnp.cos(theta))


def _partial_theta_theta_E(x, y, theta, pher_points, pher_weights, sigma_pher, beta):
    g = compute_weighted_pheromone_gradient(
        pher_points, pher_weights, x, y, sigma_pher
    )
    return beta * (g[0] * jnp.cos(theta) + g[1] * jnp.sin(theta))


# =============================================================================
# Initialization: sample a batch of agents near the source
# =============================================================================

def make_init_fn(point_a, point_b, batch_size, init_noise=0.001):
    """Build a batched initial-state sampler for the forward pass."""
    def init_fn(key):
        keys = random.split(key, batch_size)

        def single_agent(k):
            dxdy = random.normal(k, (2,)) * init_noise
            x0 = point_a[0] + dxdy[0]
            y0 = point_a[1] + dxdy[1]
            theta0 = jnp.arctan2(point_b[1] - y0, point_b[0] - x0)
            return jnp.array([x0, y0, theta0])

        return vmap(single_agent)(keys)

    return init_fn


# =============================================================================
# Discrete integration of Eq. 4 for one arc-length step
# =============================================================================
#
# We package the heading update in one place so both the forward pass and
# the controlled backward pass use the same discretization.

def _step_heading(theta, u_pher, u_goal, u_ctrl, nu_val, noise,
                  dt, beta, D_theta, eps_theta, goal_gain):
    """Euler step of Eq. 4 for theta, plus a goal-attraction stabilizer.

    Starting from the original-time SDE and reparametrizing to arc length s
    (via dt = nu * ds), Ito's rule turns the Brownian noise increment dW(t)
    into sqrt(nu) * dW_tilde(s).  This gives the simulation-form of Eq. 4:

        d theta = (nu/eps_theta) * (u_pher + u_ctrl) * ds
                 + sqrt(nu) * sqrt(2 D_theta / eps_theta) * dW_tilde(s)

    (The paper writes the same equation more compactly with a factor of nu
    outside the whole bracket; the sqrt on the noise is the Ito-consistent
    form when the noise is expressed in arc-length coordinates.)

    The `goal_gain * u_goal * ds` term is a numerical stabilizer outside
    Eq. 4; it keeps agents oriented toward the goal in early cycles when
    the pheromone field is empty and the trail-following term is silent.
    """
    diffusion = jnp.sqrt(nu_val) * jnp.sqrt(2.0 * D_theta / eps_theta)
    return (
        theta
        + (nu_val / eps_theta) * (u_pher + u_ctrl) * dt
        + goal_gain * u_goal * dt
        + diffusion * jnp.sqrt(dt) * noise
    )


# =============================================================================
# Stage 1: forward pass  (source -> target)  |  Eq. 4 + Eq. 2
# =============================================================================

def simulate_forward_batch(key, init_states, pher_points, pher_weights,
                           sigma_pher, dt, num_steps, goal,
                           nu_fn=smooth_piecewise_nu,
                           beta=1.0, D_theta=0.5, eps_theta=1.0,
                           goal_gain=10.0):
    """Integrate Eq. 4 forward from s=0 (source) to s=1 (target).

    Heading update per step (see `_step_heading`):
        - trail-following torque u_pher (Eq. 2)
        - goal-attraction stabilizer u_goal (numerical, outside Eq. 4)
        - angular noise proportional to nu * sqrt(2 D_theta / eps_theta)

    No control term is applied here (that is Stage 3).

    Returns
    -------
    trajs : (num_steps, batch_size, 3) array
    """
    batch_size = init_states.shape[0]
    zero_ctrl = jnp.zeros(batch_size)

    def step_fn(carry, _):
        key, state = carry
        key, subkey = random.split(key)
        x, y, theta = state[:, 0], state[:, 1], state[:, 2]

        # Eq. 2: trail-following torque, projected onto the heading normal
        grad_phi = vmap(
            lambda x_, y_: compute_weighted_pheromone_gradient(
                pher_points, pher_weights, x_, y_, sigma_pher
            )
        )(x, y)
        heading_normal = jnp.stack([-jnp.sin(theta), jnp.cos(theta)], axis=1)
        u_pher = beta * jnp.sum(grad_phi * heading_normal, axis=1)

        # Goal-attraction stabilizer (numerical, not in Eq. 4)
        theta_to_goal = jnp.arctan2(goal[1] - y, goal[0] - x)
        u_goal = jnp.arctan2(jnp.sin(theta_to_goal - theta),
                             jnp.cos(theta_to_goal - theta))

        # Angular noise (Ito-Euler, batched)
        noise = random.normal(subkey, (batch_size,))

        nu_val = vmap(nu_fn)(x, y)
        theta_new = _step_heading(theta, u_pher, u_goal, zero_ctrl, nu_val,
                                  noise, dt, beta, D_theta, eps_theta,
                                  goal_gain)

        # Unit-speed position update in reparametrized coordinates
        x_new = x + dt * jnp.cos(theta_new)
        y_new = y + dt * jnp.sin(theta_new)
        state_new = jnp.stack([x_new, y_new, theta_new], axis=1)
        return (key, state_new), state_new

    (_, _), trajs = lax.scan(step_fn, (key, init_states), None,
                             length=num_steps)
    return trajs


# =============================================================================
# Stage 2: adjoint sweep (Eq. 6, backward in arc length s)
# =============================================================================

def integrate_costate(trajs, pher_points, pher_weights, sigma_pher, dt,
                      nu_fn=smooth_piecewise_nu, beta=1.0, eps_theta=1.0,
                      terminal_grad_psi=None):
    """Integrate Eq. 6 BACKWARD in s along each forward trajectory.

    ODEs (paper Eq. 6):

        mu'(s)              = grad_nu + grad(nu * dE/dtheta)
        eps_theta * Gamma'(s) = nu * d^2E/dtheta^2 * Gamma + mu . (-sin, cos)

    Terminal conditions:  mu(1) = -grad Psi,  Gamma(1) = 0.
    For Psi trivial (default here), mu(1) = 0.

    Implementation.  We integrate forward in tau = 1 - s.  In that variable
    the ODE picks up a global minus sign, and the initial condition (at
    tau = 0) is exactly the paper's terminal condition (at s = 1).  We scan
    through `trajs` in REVERSE order so that at scan step k the trajectory
    state comes from s = 1 - k*dt, matching the point where dmu/dtau and
    dGamma/dtau are evaluated.  The initial state is prepended so that
    `lambda_traj[-1]` reflects the terminal condition exactly.

    Returns
    -------
    lambda_traj : (num_steps, batch_size, 3) array
        Column 0, 1: mu(s).  Column 2: Gamma(s).
        `lambda_traj[i]` is aligned with `trajs[i]` (same s).
    """
    T, B, _ = trajs.shape

    if terminal_grad_psi is None:
        mu_terminal = jnp.zeros((B, 2))
    else:
        mu_terminal = -terminal_grad_psi
    Gamma_terminal = jnp.zeros((B, 1))
    lambda_init = jnp.concatenate([mu_terminal, Gamma_terminal], axis=1)

    # Per-agent nu(x, y) and its spatial gradient, batched via vmap.
    v_nu = vmap(nu_fn)
    v_grad_nu = vmap(grad(nu_fn, argnums=(0, 1)))

    # nu * dE/dtheta as a function of (x, y) at fixed theta and fixed
    # pheromone field.  Its spatial gradient enters mu'.
    def nu_partial_theta_E(x_, y_, theta_):
        return nu_fn(x_, y_) * _partial_theta_E(
            x_, y_, theta_, pher_points, pher_weights, sigma_pher, beta
        )

    v_grad_nu_pt_E = vmap(
        grad(nu_partial_theta_E, argnums=(0, 1)),
        in_axes=(0, 0, 0),
    )

    v_partial_theta_theta_E = vmap(
        lambda x_, y_, theta_: _partial_theta_theta_E(
            x_, y_, theta_, pher_points, pher_weights, sigma_pher, beta
        ),
        in_axes=(0, 0, 0),
    )

    def step(carry, state_t):
        mu = carry[:, :2]
        Gamma = carry[:, 2]
        x, y, theta = state_t[:, 0], state_t[:, 1], state_t[:, 2]

        # mu'(s) = grad nu + grad(nu * dE/dtheta)
        dnu_dx, dnu_dy = v_grad_nu(x, y)
        dNuE_dx, dNuE_dy = v_grad_nu_pt_E(x, y, theta)
        mu_prime = jnp.stack([dnu_dx + dNuE_dx, dnu_dy + dNuE_dy], axis=1)

        # eps_theta * Gamma'(s) = nu * d^2E/dtheta^2 * Gamma + mu . heading_perp
        nu_val = v_nu(x, y)
        pt2_E = v_partial_theta_theta_E(x, y, theta)
        heading_perp = jnp.stack([-jnp.sin(theta), jnp.cos(theta)], axis=1)
        Gamma_prime = (nu_val * pt2_E * Gamma
                       + jnp.sum(mu * heading_perp, axis=1)) / eps_theta

        # Backward integration in tau = 1 - s.  With `mu_prime` and
        # `Gamma_prime` equal to the paper's Eq. 6 RHS interpreted as d/ds,
        # the change of variable gives d/dtau = -d/ds:
        new_mu = mu - mu_prime * dt
        new_Gamma = Gamma - Gamma_prime * dt

        new_carry = jnp.concatenate([new_mu, new_Gamma[:, None]], axis=1)
        return new_carry, new_carry

    # Feed trajectory states in reverse; the last one (trajs[0]) is unused
    # because we already have T-1 dt-steps to cover s=1 down to s=dt.
    trajs_rev = trajs[::-1][:-1]

    _, lambda_after = lax.scan(step, lambda_init, trajs_rev)
    # Prepend the terminal condition so length matches T and the terminal
    # condition sits at the correct index (s = 1) after the reversal below.
    lambda_traj_rev = jnp.concatenate([lambda_init[None], lambda_after], axis=0)
    lambda_traj = lambda_traj_rev[::-1]  # index 0 == s=0, index T-1 == s=1

    # Sign convention.  The scan integrates a direct augmented-Lagrangian
    # form whose costate has the opposite sign to the paper's Gamma, mu.
    # Negating on return aligns with the manuscript (Eq. 6 as written,
    # Eq. 7 in its literal form omega_ctrl = -Gamma/gamma).  Terminal
    # conditions Gamma(1)=0 and mu(1)=0 are invariant under this sign.
    return -lambda_traj


# =============================================================================
# Stage 3: controlled backward pass (target -> source)  |  Eq. 7
# =============================================================================

def simulate_controlled_backward_pass(key, final_states, lambda_traj,
                                      pher_points, pher_weights, sigma_pher,
                                      num_steps, dt, goal,
                                      nu_fn=smooth_piecewise_nu,
                                      beta=1.0, D_theta=0.5, eps_theta=1.0,
                                      goal_gain=10.0):
    """Return agents from target back to source under Eq. 7.

    Feedback law:
        omega_ctrl(s) = - Gamma(s) / gamma,   gamma = beta * D_theta

    The `lambda_traj` argument is the costate from Stage 2; only its
    theta-component (Gamma) enters Eq. 7.  `final_states` is the
    end-of-forward-pass state with heading rotated by pi so that agents
    face the source, and `goal` is set to `point_a`.

    Physical parameters (beta, D_theta, eps_theta) must match Stage 1 so
    the FDR gamma = beta*D_theta holds self-consistently.

    Returns
    -------
    trajs : (num_steps, batch_size, 3) array
    """
    batch_size = final_states.shape[0]
    step_keys = random.split(key, num_steps)
    gamma = beta * D_theta  # Eq. 7 gain (FDR)

    def step_fn(state, inputs):
        lambda_t, key_t = inputs
        x, y, theta = state[:, 0], state[:, 1], state[:, 2]

        # Eq. 2: trail-following torque
        grad_phi = vmap(
            lambda x_, y_: compute_weighted_pheromone_gradient(
                pher_points, pher_weights, x_, y_, sigma_pher
            )
        )(x, y)
        heading_normal = jnp.stack([-jnp.sin(theta), jnp.cos(theta)], axis=1)
        u_pher = beta * jnp.sum(grad_phi * heading_normal, axis=1)

        # Goal-attraction stabilizer (numerical, not in Eq. 4)
        theta_to_goal = jnp.arctan2(goal[1] - y, goal[0] - x)
        u_goal = jnp.arctan2(jnp.sin(theta_to_goal - theta),
                             jnp.cos(theta_to_goal - theta))

        # Eq. 7: optimal control feedback  omega_ctrl(s) = -Gamma(s)/gamma
        Gamma = lambda_t[:, 2]
        u_ctrl = -Gamma / gamma

        noise = random.normal(key_t, (batch_size,))
        nu_val = vmap(nu_fn)(x, y)
        theta_new = _step_heading(theta, u_pher, u_goal, u_ctrl, nu_val,
                                  noise, dt, beta, D_theta, eps_theta,
                                  goal_gain)

        x_new = x + dt * jnp.cos(theta_new)
        y_new = y + dt * jnp.sin(theta_new)
        new_state = jnp.stack([x_new, y_new, theta_new], axis=1)
        return new_state, new_state

    _, trajs = lax.scan(step_fn, final_states, (lambda_traj, step_keys))
    return trajs


# =============================================================================
# Stage 4: pheromone deposition (recency-weighted resampling)
# =============================================================================

def downsample_recent_weighted_trajectories(all_backward_trajectories,
                                            num_trajs_to_sample,
                                            weight=1.0, key=None, alpha=0.5):
    """Build the next cycle's pheromone field from past controlled returns.

    Trajectories from all completed cycles are pooled; each is assigned a
    sampling weight (cycle_index + 1) ** alpha, so more recent cycles
    contribute more heavily but older cycles are not fully forgotten.  A
    fixed number `num_trajs_to_sample` of trajectories is drawn without
    replacement, and their (x, y) samples become the point cloud that
    parametrizes phi for the next forward pass.
    """
    if key is None:
        raise ValueError("Must provide a PRNG key.")

    batch_per_cycle = [traj.shape[1] for traj in all_backward_trajectories]
    total_trajs = sum(batch_per_cycle)
    pooled = jnp.concatenate(all_backward_trajectories, axis=1)

    cycle_ids = jnp.concatenate(
        [jnp.full((b,), i) for i, b in enumerate(batch_per_cycle)]
    )
    probs = (cycle_ids + 1) ** alpha
    probs = probs / probs.sum()

    num_to_draw = jnp.minimum(num_trajs_to_sample, total_trajs)
    idx = random.choice(key, total_trajs, shape=(num_to_draw,),
                        p=probs, replace=False)

    sampled = pooled[:, idx, :2]
    pheromone_points = sampled.reshape(-1, 2)
    pheromone_weights = jnp.full((pheromone_points.shape[0],), weight)
    return pheromone_points, pheromone_weights


# =============================================================================
# Algorithm 1: the iterative forward-backward loop
# =============================================================================

def run_apic_loop(num_cycles, key, init_fn, point_a, point_b,
                  dt, num_steps, pher_sigma,
                  beta=1.0, D_theta=0.5, eps_theta=1.0, goal_gain=10.0,
                  nu_fn=smooth_piecewise_nu,
                  num_trajs_to_sample=200):
    """Run `num_cycles` stigmergic cycles of Algorithm 1.

    Each cycle:
        1. sample a batch of agents at the source
        2. Stage 1: forward pass from source to target (Eq. 4 + Eq. 2)
        3. Stage 2: adjoint sweep, backward in arc length (Eq. 6)
        4. Stage 3: controlled return from target to source (Eq. 7)
        5. Stage 4: recency-weighted resampling to update the pheromone field

    Physical parameters (beta, D_theta, eps_theta) shape both Stages 1 and 3
    identically; the fluctuation-dissipation relation gamma = beta*D_theta
    (Eq. 7) is enforced internally by Stage 3.

    Returns
    -------
    all_forward : list of (T, B, 3) arrays, one per cycle
    all_backward : list of (T, B, 3) arrays, one per cycle
    all_pher_points : (N, 2) final pheromone locations
    all_pher_weights : (N,) final pheromone weights
    """
    all_pher_points = jnp.empty((0, 2))
    all_pher_weights = jnp.empty((0,))
    all_forward = []
    all_backward = []

    for _ in trange(num_cycles, desc="APIC cycle"):
        key, k_init, k_fwd, k_bwd, k_phen = random.split(key, 5)

        # Fresh batch of agents at the source
        init_states = init_fn(k_init)

        # --- Stage 1: forward pass  (Eq. 4 + Eq. 2)
        trajs_forward = simulate_forward_batch(
            k_fwd, init_states, all_pher_points, all_pher_weights,
            pher_sigma, dt, num_steps, point_b,
            nu_fn=nu_fn,
            beta=beta, D_theta=D_theta, eps_theta=eps_theta,
            goal_gain=goal_gain,
        )
        all_forward.append(trajs_forward)

        # --- Stage 2: adjoint sweep  (Eq. 6, backward in s)
        lambda_traj = integrate_costate(
            trajs_forward, all_pher_points, all_pher_weights,
            pher_sigma, dt,
            nu_fn=nu_fn, beta=beta, eps_theta=eps_theta,
        )

        # --- Stage 3: controlled backward pass  (Eq. 7)
        # Take the end-of-forward state and flip the heading by pi so agents
        # face back toward the source before starting the return integration.
        final_states = trajs_forward[-1]
        x, y, theta = final_states[:, 0], final_states[:, 1], final_states[:, 2]
        theta_back = jnp.mod(theta + jnp.pi, 2.0 * jnp.pi)
        final_states_rot = jnp.stack([x, y, theta_back], axis=1)

        trajs_backward = simulate_controlled_backward_pass(
            k_bwd, final_states_rot, lambda_traj,
            all_pher_points, all_pher_weights, pher_sigma,
            num_steps, dt, point_a,
            nu_fn=nu_fn,
            beta=beta, D_theta=D_theta, eps_theta=eps_theta,
            goal_gain=goal_gain,
        )
        all_backward.append(trajs_backward)

        # --- Stage 4: pheromone update
        all_pher_points, all_pher_weights = downsample_recent_weighted_trajectories(
            all_backward_trajectories=all_backward,
            num_trajs_to_sample=num_trajs_to_sample,
            weight=1.0,
            key=k_phen,
        )

    return all_forward, all_backward, all_pher_points, all_pher_weights


# =============================================================================
# Snell-Descartes diagnostic: sin theta_1 / sin theta_2 per trajectory
# =============================================================================

def arc_length(x, y):
    """Total polyline length of the (x, y) samples."""
    dx = jnp.diff(x)
    dy = jnp.diff(y)
    return jnp.sum(jnp.sqrt(dx**2 + dy**2))


def compute_sin_ratio(trajs, boundary=0.5):
    """Snell ratio sin(theta_1)/sin(theta_2) for each trajectory crossing.

    Splits each trajectory at its first upward crossing of y = boundary,
    computes the chord angle to the vertical on either side (via a chord
    over its arc length), and returns their sine ratio.  Trajectories that
    do not cross, or that cross at the very endpoints, are marked NaN.
    """
    T, B, _ = trajs.shape
    ratios = []
    for i in range(B):
        traj = trajs[:, i]
        x, y = traj[:, 0], traj[:, 1]
        crossing_idx = int(jnp.argmax(y > boundary))
        if y[crossing_idx] <= boundary or crossing_idx in (0, T - 1):
            ratios.append(jnp.nan)
            continue

        x1, y1 = x[:crossing_idx + 1], y[:crossing_idx + 1]
        len1 = arc_length(x1, y1)
        cos1 = (y1[-1] - y1[0]) / len1
        sin1 = jnp.sqrt(1.0 - cos1**2)

        x2, y2 = x[crossing_idx:], y[crossing_idx:]
        len2 = arc_length(x2, y2)
        cos2 = (y2[-1] - y2[0]) / len2
        sin2 = jnp.sqrt(1.0 - cos2**2)

        ratios.append(sin1 / sin2)

    return jnp.array(ratios)
