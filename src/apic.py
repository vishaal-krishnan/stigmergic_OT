"""
Adjoint Path Integral Control (APIC) for stigmergic optimal transport.

Implements Algorithm 1 from the main text: the iterative forward-backward
optimization loop used for all Snell-Descartes results in the paper.

Each stigmergic cycle consists of:
    (1-2) Forward pass:  simulate_forward_batch
    (3)   Adjoint pass:  integrate_costate
    (4)   Controlled backward pass: simulate_controlled_backward_pass
    (5)   Pheromone update: downsample_recent_weighted_trajectories

The entry point `run_apic_loop` executes the full iteration.
"""

import jax
import jax.numpy as jnp
from jax import grad, vmap, lax, random
from tqdm import trange


# ---------------------------------------------------------------------------
# Refractive index field
# ---------------------------------------------------------------------------

def smooth_piecewise_nu(x, y, base=1.0, jump=10.0, steep=100.0, boundary=0.5):
    """Smooth step-like refractive index nu(x,y), transitioning at y=boundary."""
    s = 1.0 / (1.0 + jnp.exp(-steep * (y - boundary)))
    return base * (1 - s) + jump * s


# ---------------------------------------------------------------------------
# Pheromone field (Gaussian kernel density estimate)
# ---------------------------------------------------------------------------

def compute_weighted_pheromone_gradient(points, weights, x, y, sigma):
    """Gradient of log pheromone field at (x, y)."""
    def eval_pher(xx, yy):
        dx = xx - points[:, 0]
        dy = yy - points[:, 1]
        vals = jnp.exp(-(dx**2 + dy**2) / (2 * sigma**2))
        return jnp.log(0.01 + jnp.sum(vals * weights))

    gx = grad(lambda xx: eval_pher(xx, y))(x)
    gy = grad(lambda yy: eval_pher(x, yy))(y)
    return jnp.array([gx, gy])


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def make_init_fn(point_a, point_b, batch_size, init_noise=0.001):
    """Return a function that samples batch_size initial agent states near point_a,
    each heading toward point_b."""
    def init_fn(key):
        keys = random.split(key, batch_size)
        def single_agent(k):
            dxdy = random.normal(k, (2,)) * init_noise
            x0 = point_a[0] + dxdy[0]
            y0 = point_a[1] + dxdy[1]
            theta0 = jnp.arctan2(point_b[1] - y0, point_b[0] - x0)
            return jnp.array([x0, y0, theta0])
        return jax.vmap(single_agent)(keys)
    return init_fn


# ---------------------------------------------------------------------------
# Step 1-2: Forward pass
# ---------------------------------------------------------------------------

def simulate_forward_batch(key, init_states, pher_points, pher_weights,
                           sigma_pher, dt, num_steps, sigma_noise, goal):
    """Integrate the arc-length-reparametrized Langevin dynamics forward
    from source to target for a batch of agents under the current pheromone
    field. Applies trail-following (Eq. 2) and stochastic angular noise."""
    batch_size = init_states.shape[0]

    def step_fn(carry, _):
        key, state = carry
        key, subkey = random.split(key)
        x, y, theta = state[:, 0], state[:, 1], state[:, 2]
        speed = 1.0

        grad_pher = vmap(lambda x_, y_: compute_weighted_pheromone_gradient(
            pher_points, pher_weights, x_, y_, sigma_pher))(x, y)
        n_hat = jnp.stack([-jnp.sin(theta), jnp.cos(theta)], axis=1)
        u_pher = jnp.sum(grad_pher * n_hat, axis=1)

        desired_theta_goal = jnp.arctan2(goal[1] - y, goal[0] - x)
        u_goal = jnp.arctan2(jnp.sin(desired_theta_goal - theta),
                             jnp.cos(desired_theta_goal - theta))

        noise = sigma_noise * random.normal(subkey, (batch_size,))
        theta_new = (theta + 1.0 * dt * u_pher + 10.0 * dt * u_goal
                     + jnp.sqrt(dt) * noise)

        x_new = x + dt * speed * jnp.cos(theta_new)
        y_new = y + dt * speed * jnp.sin(theta_new)
        state_new = jnp.stack([x_new, y_new, theta_new], axis=1)
        return (key, state_new), state_new

    (_, _), trajs = lax.scan(step_fn, (key, init_states), None, length=num_steps)
    return trajs  # (T, B, 3)


# ---------------------------------------------------------------------------
# Step 3: Adjoint / costate backward integration
# ---------------------------------------------------------------------------

def integrate_costate(trajs, nu_fn, dt):
    """Integrate the adjoint equations (Eq. 6) backward along each forward
    trajectory, yielding the costate Gamma(s)."""
    num_steps, batch_size, _ = trajs.shape

    dnu_dx = jax.vmap(grad(lambda x_, y_: nu_fn(x_, y_)), in_axes=(0, 0))
    dnu_dy = jax.vmap(grad(lambda y_, x_: nu_fn(x_, y_)), in_axes=(0, 0))

    def adjoint_dynamics(lambda_t, state_t):
        x, y, theta = state_t[:, 0], state_t[:, 1], state_t[:, 2]
        grad_nu = jnp.stack([dnu_dx(x, y), dnu_dy(x, y)], axis=1)
        fx_theta = jnp.stack([-jnp.sin(theta), jnp.cos(theta)], axis=1)
        dH_dtheta = jnp.sum(lambda_t[:, :2] * fx_theta, axis=1)
        lambda_dot = -jnp.stack([grad_nu[:, 0], grad_nu[:, 1], dH_dtheta], axis=1)
        return lambda_t + lambda_dot * dt

    def scan_fn(carry, state_t):
        lam = adjoint_dynamics(carry, state_t)
        return lam, lam

    lambda_T = jnp.zeros((batch_size, 3))
    _, lambda_traj = lax.scan(scan_fn, lambda_T, trajs)
    return lambda_traj[::-1]


# ---------------------------------------------------------------------------
# Step 4: Controlled backward pass
# ---------------------------------------------------------------------------

def simulate_controlled_backward_pass(key, final_states, lambda_traj,
                                      pher_points, pher_weights, sigma_pher,
                                      num_steps, dt, sigma_noise, goal):
    """Generate controlled return trajectories from target back to source
    using the optimal control omega_ctrl = -Gamma/gamma (Eq. 7)."""
    batch_size = final_states.shape[0]
    keys = random.split(key, num_steps)

    def step_fn(state, inputs):
        lambda_t, key_t = inputs
        x, y, theta = state[:, 0], state[:, 1], state[:, 2]
        speed = 1.0

        # Control from the costate (theta component of lambda)
        l_theta = lambda_t[:, 2]
        u_control = l_theta

        grad_pher = vmap(lambda x_, y_: compute_weighted_pheromone_gradient(
            pher_points, pher_weights, x_, y_, sigma_pher))(x, y)
        n_hat = jnp.stack([-jnp.sin(theta), jnp.cos(theta)], axis=1)
        u_pher = jnp.sum(grad_pher * n_hat, axis=1)

        desired_theta_goal = jnp.arctan2(goal[1] - y, goal[0] - x)
        u_goal = jnp.arctan2(jnp.sin(desired_theta_goal - theta),
                             jnp.cos(desired_theta_goal - theta))

        noise = sigma_noise * random.normal(key_t, shape=(batch_size,))
        theta_new = (theta + 1.0 * dt * u_pher + 10.0 * dt * u_goal
                     + 1.0 * dt * u_control + jnp.sqrt(dt) * noise)
        x_new = x + dt * speed * jnp.cos(theta_new)
        y_new = y + dt * speed * jnp.sin(theta_new)
        new_state = jnp.stack([x_new, y_new, theta_new], axis=1)
        return new_state, new_state

    _, trajs = lax.scan(step_fn, final_states, (lambda_traj, keys))
    return trajs


# ---------------------------------------------------------------------------
# Step 5: Pheromone deposition (resample backward trajectories)
# ---------------------------------------------------------------------------

def downsample_recent_weighted_trajectories(all_backward_trajectories,
                                            num_trajs_to_sample,
                                            weight=1.0, key=None, alpha=0.5):
    """Resample trajectories across all cycles, favoring more recent cycles
    (alpha controls the recency bias). Sampled points become the new
    pheromone field."""
    if key is None:
        raise ValueError("Must provide PRNG key.")

    B_per_cycle = [traj.shape[1] for traj in all_backward_trajectories]
    B_total = sum(B_per_cycle)
    trajs_all = jnp.concatenate(all_backward_trajectories, axis=1)  # (T, B_total, 3)

    cycle_ids = jnp.concatenate(
        [jnp.full((B,), i) for i, B in enumerate(B_per_cycle)]
    )
    probs = (cycle_ids + 1) ** alpha
    probs = probs / probs.sum()

    num_trajs_to_sample = jnp.minimum(num_trajs_to_sample, B_total)
    idx = random.choice(key, B_total, shape=(num_trajs_to_sample,),
                        p=probs, replace=False)

    sampled = trajs_all[:, idx, :2]  # (T, N, 2)
    pheromone_points = sampled.reshape(-1, 2)
    pheromone_weights = jnp.full((pheromone_points.shape[0],), weight)
    return pheromone_points, pheromone_weights


# ---------------------------------------------------------------------------
# Algorithm 1: the full iterative forward-backward loop
# ---------------------------------------------------------------------------

def run_apic_loop(num_cycles, key, init_fn, point_a, point_b, pher_sigma,
                  forward_params, backward_params, nu_fn=smooth_piecewise_nu,
                  num_trajs_to_sample=200):
    """Execute Algorithm 1 over `num_cycles` stigmergic cycles.

    Each cycle: forward sampling -> adjoint backward -> controlled return
    -> pheromone update.

    Returns
    -------
    all_forward : list of (T, B, 3) arrays, one per cycle
    all_backward : list of (T, B, 3) arrays, one per cycle
    all_pher_points : (N, 2) final pheromone field locations
    all_pher_weights : (N,) corresponding weights
    """
    all_pher_points = jnp.empty((0, 2))
    all_pher_weights = jnp.empty((0,))
    all_forward = []
    all_backward = []

    for _ in trange(num_cycles, desc="APIC cycle"):
        key, k_init, k_fwd, k_bwd, k_phen = random.split(key, 5)

        # Initialize agents near the source
        init_states = init_fn(k_init)

        # Step 1-2: Forward pass (source -> target)
        trajs_forward = simulate_forward_batch(
            k_fwd, init_states, all_pher_points, all_pher_weights,
            pher_sigma, forward_params["dt"], forward_params["num_steps"],
            forward_params["sigma_noise"], point_b
        )
        all_forward.append(trajs_forward)

        # Step 3: Adjoint integration (backward in arc-length s)
        lambda_traj = integrate_costate(trajs_forward, nu_fn,
                                        forward_params["dt"])

        # Step 4: Controlled backward pass (target -> source)
        final_states = trajs_forward[-1]
        x, y, theta = final_states[:, 0], final_states[:, 1], final_states[:, 2]
        theta_flipped = jnp.mod(theta + jnp.pi, 2 * jnp.pi)
        final_states_rotated = jnp.stack([x, y, theta_flipped], axis=1)

        trajs_backward = simulate_controlled_backward_pass(
            k_bwd, final_states_rotated, lambda_traj,
            all_pher_points, all_pher_weights, pher_sigma,
            backward_params["num_steps"], backward_params["dt"],
            backward_params["sigma_noise"], point_a
        )
        all_backward.append(trajs_backward)

        # Step 5: Pheromone update from backward trajectories
        all_pher_points, all_pher_weights = downsample_recent_weighted_trajectories(
            all_backward_trajectories=all_backward,
            num_trajs_to_sample=num_trajs_to_sample,
            weight=1.0,
            key=k_phen,
        )

    return all_forward, all_backward, all_pher_points, all_pher_weights


# ---------------------------------------------------------------------------
# Snell ratio diagnostic
# ---------------------------------------------------------------------------

def arc_length(x, y):
    dx = jnp.diff(x)
    dy = jnp.diff(y)
    return jnp.sum(jnp.sqrt(dx**2 + dy**2))


def compute_sin_ratio(trajs, boundary=0.5):
    """Compute sin(incident)/sin(refracted) for each trajectory crossing y=boundary."""
    T, B, _ = trajs.shape
    ratios = []
    for i in range(B):
        traj = trajs[:, i]
        x, y = traj[:, 0], traj[:, 1]
        crossing_idx = int(jnp.argmax(y > boundary))
        if y[crossing_idx] <= boundary or crossing_idx == 0 or crossing_idx == T - 1:
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
