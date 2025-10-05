"""
Inhomogeneous Media Optimization Implementation

This module implements path optimization in inhomogeneous media with 
varying refractive indices (analogous to light refraction).
"""

import jax
import jax.numpy as jnp
from jax import grad, vmap, random, lax
import numpy as np
import matplotlib.pyplot as plt

def smooth_piecewise_nu(x, y, base=1.0, jump=10.0, steep=100.0, boundary=0.5):
    """Smooth refractive index field with transition at boundary."""
    s = 1.0 / (1.0 + jnp.exp(-steep * (y - boundary)))
    return base * (1 - s) + jump * s

def grad_ln_nu(x, y, base=1.0, jump=10.0, steep=100.0, boundary=0.5):
    """Gradient of log refractive index."""
    s = 1.0 / (1.0 + jnp.exp(-steep * (y - boundary)))
    nu_val = smooth_piecewise_nu(x, y, base, jump, steep, boundary)
    ds_dy = steep * s * (1 - s)
    dln_nu_dy = (jump - base) * ds_dy / nu_val
    return jnp.array([0.0, dln_nu_dy])

def compute_weighted_pheromone_gradient(points, weights, x, y, sigma):
    """Compute weighted pheromone gradient."""
    def eval_pher(xx, yy):
        dx = xx - points[:, 0]
        dy = yy - points[:, 1]
        vals = jnp.exp(-(dx**2 + dy**2) / (2 * sigma**2))
        return jnp.log(0.01 + jnp.sum(vals * weights))

    gx = grad(lambda xx: eval_pher(xx, y))(x)
    gy = grad(lambda yy: eval_pher(x, yy))(y)
    return jnp.array([gx, gy])

def update_agent_state_inhomogeneous(state, pher_points, pher_weights, sigma_pher, dt, 
                                   beta, alpha_nu, gamma_goal, noise_std, key, point_b):
    """Update agent state in inhomogeneous media with pheromone and refractive index effects."""
    x, y, theta = state
    
    # Pheromone gradient
    grad_pher = compute_weighted_pheromone_gradient(pher_points, pher_weights, x, y, sigma_pher)
    n_hat = jnp.array([-jnp.sin(theta), jnp.cos(theta)])
    u_pher = jnp.dot(grad_pher, n_hat)
    
    # Refractive index gradient effect
    nu_val = smooth_piecewise_nu(x, y)
    grad_nu = grad_ln_nu(x, y)
    u_nu = alpha_nu * jnp.dot(grad_nu, n_hat) / nu_val
    
    # Goal attraction
    goal_vec = jnp.array([point_b[0] - x, point_b[1] - y])
    goal_dist = jnp.linalg.norm(goal_vec)
    u_goal = gamma_goal * jnp.dot(goal_vec, n_hat) / (goal_dist + 1e-6)
    
    # Add noise
    key, subkey = random.split(key)
    noise = noise_std * random.normal(subkey)
    
    # Update angle
    theta_new = theta + dt * (beta * u_pher + u_nu + u_goal + noise)
    
    # Update position (speed affected by refractive index)
    speed = 1.0 / nu_val  # Speed inversely proportional to refractive index
    x_new = x + dt * speed * jnp.cos(theta_new)
    y_new = y + dt * speed * jnp.sin(theta_new)
    
    return jnp.array([x_new, y_new, theta_new]), key

def simulate_inhomogeneous_optimization(pher_points, pher_weights, num_steps, dt, sigma_pher,
                                      beta, alpha_nu, gamma_goal, noise_std, key, point_a, point_b):
    """Simulate agent optimization in inhomogeneous media."""
    # Initialize agent
    key, subkey = random.split(key)
    dxdy = random.normal(subkey, (2,)) * 0.01
    x0 = point_a[0] + dxdy[0]
    y0 = point_a[1] + dxdy[1]
    theta0 = jnp.arctan2(point_b[1] - y0, point_b[0] - x0)
    state = jnp.array([x0, y0, theta0])
    
    def body_fn(carry, _):
        st, ky = carry
        new_st, new_ky = update_agent_state_inhomogeneous(
            st, pher_points, pher_weights, sigma_pher, dt, beta, alpha_nu, 
            gamma_goal, noise_std, ky, point_b
        )
        return (new_st, new_ky), new_st[:2]
    
    (final_state, _), traj = lax.scan(body_fn, (state, key), jnp.arange(num_steps))
    return jnp.vstack([state[:2], traj])

def compute_optical_path_length(trajectory, dt):
    """Compute optical path length (integral of nu along path)."""
    x, y = trajectory[:, 0], trajectory[:, 1]
    nu_vals = vmap(smooth_piecewise_nu)(x, y)
    
    # Compute path segments
    dx = jnp.diff(x)
    dy = jnp.diff(y)
    ds = jnp.sqrt(dx**2 + dy**2)
    
    # Optical path length = sum of nu * ds
    return jnp.sum(nu_vals[:-1] * ds)

def snell_optimal_path_length(point_a, point_b, y_interface, nu1, nu2):
    """Compute optimal path length according to Snell's law."""
    xa, ya = point_a
    xb, yb = point_b
    
    def time_cost(xi):
        L1 = jnp.sqrt((xi - xa)**2 + (y_interface - ya)**2)
        L2 = jnp.sqrt((xb - xi)**2 + (yb - y_interface)**2)
        return L1 * nu1 + L2 * nu2
    
    # Find optimal crossing point
    xi_vals = jnp.linspace(0, 1, 1000)
    costs = vmap(time_cost)(xi_vals)
    optimal_xi = xi_vals[jnp.argmin(costs)]
    
    return time_cost(optimal_xi), optimal_xi

def run_inhomogeneous_optimization_experiment():
    """Run a complete inhomogeneous media optimization experiment."""
    # Parameters
    point_a = jnp.array([0.0, 0.0])
    point_b = jnp.array([1.0, 1.0])
    num_steps = 200
    dt = 0.01
    sigma_pher = 0.05
    beta = 5.0
    alpha_nu = 5.0
    gamma_goal = 1.0
    noise_std = 0.2
    
    # Refractive index parameters
    base_nu, jump_nu, y_interface = 1.0, 10.0, 0.5
    
    # Create initial pheromone trail (straight line)
    t = jnp.linspace(0, 1, num_steps)
    initial_trail = jnp.stack([
        point_a[0] + t * (point_b[0] - point_a[0]),
        point_a[1] + t * (point_b[1] - point_a[1])
    ], axis=1)
    
    pher_points = initial_trail
    pher_weights = jnp.ones(pher_points.shape[0])
    
    # Run optimization
    key = random.PRNGKey(42)
    optimized_trajectory = simulate_inhomogeneous_optimization(
        pher_points, pher_weights, num_steps, dt, sigma_pher, beta, alpha_nu, 
        gamma_goal, noise_std, key, point_a, point_b
    )
    
    # Compute performance metrics
    optical_length = compute_optical_path_length(optimized_trajectory, dt)
    snell_optimal, optimal_xi = snell_optimal_path_length(point_a, point_b, y_interface, base_nu, jump_nu)
    efficiency = snell_optimal / optical_length
    
    return {
        'initial_trail': initial_trail,
        'optimized_trajectory': optimized_trajectory,
        'optical_length': optical_length,
        'snell_optimal': snell_optimal,
        'optimal_crossing': optimal_xi,
        'efficiency': efficiency
    }

def plot_inhomogeneous_results(results, save_path=None):
    """Plot inhomogeneous media optimization results."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Trajectories with refractive index field
    ax1 = axes[0]
    
    # Create refractive index background
    x_vals = np.linspace(-0.1, 1.1, 200)
    y_vals = np.linspace(-0.1, 1.1, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    nu_field = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            nu_field[i, j] = smooth_piecewise_nu(X[i, j], Y[i, j])
    
    im = ax1.contourf(X, Y, nu_field, levels=50, cmap='RdYlBu_r', alpha=0.6)
    plt.colorbar(im, ax=ax1, label='Refractive Index ν')
    
    # Plot trajectories
    ax1.plot(results['initial_trail'][:, 0], results['initial_trail'][:, 1], 
             'k--', linewidth=2, label='Straight Path', alpha=0.5)
    ax1.plot(results['optimized_trajectory'][:, 0], results['optimized_trajectory'][:, 1], 
             'r-', linewidth=2, label='Optimized Path')
    
    # Plot Snell's law optimal crossing
    point_a = jnp.array([0.0, 0.0])
    point_b = jnp.array([1.0, 1.0])
    optimal_xi = results['optimal_crossing']
    ax1.plot([point_a[0], optimal_xi], [point_a[1], 0.5], 'g--', linewidth=2, alpha=0.7)
    ax1.plot([optimal_xi, point_b[0]], [0.5, point_b[1]], 'g--', linewidth=2, 
             label="Snell's Optimal", alpha=0.7)
    ax1.scatter([optimal_xi], [0.5], color='green', s=100, zorder=5)
    
    # Mark interface
    ax1.axhline(0.5, color='black', linestyle=':', linewidth=2, label='Interface')
    
    # Mark start and end
    ax1.scatter([point_a[0]], [point_a[1]], color='blue', s=100, label='Start', zorder=5)
    ax1.scatter([point_b[0]], [point_b[1]], color='red', s=100, label='Goal', zorder=5)
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Path Optimization in Inhomogeneous Media')
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance metrics
    ax2 = axes[1]
    
    metrics = {
        'Optimized\nPath': results['optical_length'],
        "Snell's\nOptimal": results['snell_optimal']
    }
    
    bars = ax2.bar(metrics.keys(), metrics.values(), color=['red', 'green'], alpha=0.7)
    ax2.set_ylabel('Optical Path Length')
    ax2.set_title(f"Performance Comparison\nEfficiency: {results['efficiency']:.2%}")
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

