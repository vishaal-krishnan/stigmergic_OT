"""
Trail Following Implementation

This module implements the trail following problem: given an existing trail,
how do agents follow it using pheromone gradients?
"""

import jax
import jax.numpy as jnp
from jax import grad, vmap, random, lax
import numpy as np
import matplotlib.pyplot as plt

def create_squiggly_line(point_a, point_b, num_points=120, amplitude=0.1, frequency=1.0):
    """Create a wavy trail between two points for testing trail following."""
    t = jnp.linspace(0, 1, num_points)
    x = point_a[0] + t * (point_b[0] - point_a[0])
    y = point_a[1] + t * (point_b[1] - point_a[1])
    
    # Add sinusoidal wiggles
    noise_x = amplitude * jnp.sin(2 * jnp.pi * t * frequency) * jnp.exp(-t * 2)
    noise_y = amplitude * jnp.cos(2 * jnp.pi * t * frequency) * jnp.exp(-t * 2)
    
    return jnp.stack([x + noise_x, y + noise_y], axis=1)

def gaussian_kernel(dx, dy, sigma):
    """Isotropic Gaussian kernel for pheromone field."""
    return jnp.exp(-(dx**2 + dy**2) / (2 * sigma**2))

def compute_pheromone_gradient(trajectory_points, x, y, sigma=0.05):
    """Compute pheromone gradient at position (x, y) from trail points."""
    def eval_pheromone(x_pos, y_pos):
        dx = x_pos - trajectory_points[:, 0]
        dy = y_pos - trajectory_points[:, 1]
        vals = jnp.exp(-(dx**2 + dy**2) / (2 * sigma**2))
        return jnp.log(0.01 + jnp.sum(vals))

    gx = grad(lambda xx: eval_pheromone(xx, y))(x)
    gy = grad(lambda yy: eval_pheromone(x, yy))(y)
    return jnp.array([gx, gy])

def update_agent_state_trail_following(state, trajectory_points, beta, angular_noise_std, key, sigma=0.05):
    """Update agent state to follow a given trail."""
    x, y, theta = state
    
    # Compute pheromone gradient from trail
    grad_pher = compute_pheromone_gradient(trajectory_points, x, y, sigma)
    
    # Normal vector to current heading
    tangent_vector = jnp.array([jnp.cos(theta), jnp.sin(theta)])
    normal_vector = jnp.array([-jnp.sin(theta), jnp.cos(theta)])
    
    # Angular velocity from pheromone gradient
    angular_velocity = beta * jnp.dot(grad_pher, normal_vector)
    
    # Add noise
    key, subkey = random.split(key)
    angular_velocity += angular_noise_std * random.normal(subkey)
    
    # Update heading
    theta_new = theta + angular_velocity * 0.01  # dt = 0.01
    
    # Update position
    x_new = x + jnp.cos(theta_new) * 0.01
    y_new = y + jnp.sin(theta_new) * 0.01
    
    return jnp.array([x_new, y_new, theta_new]), key

def simulate_trail_following(trajectory_points, num_steps, beta, angular_noise_std, key, 
                           point_a, point_b, sigma=0.05):
    """Simulate an agent following a given trail."""
    # Initialize agent near start point
    key, subkey = random.split(key)
    initial_position = random.normal(subkey, (2,)) * 0.01
    x0 = point_a[0] + initial_position[0]
    y0 = point_a[1] + initial_position[1]
    theta0 = jnp.arctan2(point_b[1] - y0, point_b[0] - x0)
    state = jnp.array([x0, y0, theta0])
    
    def body_fun(carry, _):
        state, key = carry
        new_state, new_key = update_agent_state_trail_following(
            state, trajectory_points, beta, angular_noise_std, key, sigma
        )
        return (new_state, new_key), new_state[:2]
    
    (_, _), trajectory = lax.scan(body_fun, (state, key), jnp.arange(num_steps))
    trajectory = jnp.vstack([state[:2][None, :], trajectory])
    
    return trajectory

def evaluate_trail_following_quality(trail, trajectory, dt=0.01):
    """Evaluate how well the agent followed the trail."""
    # Ensure both have same length
    min_len = min(trail.shape[0], trajectory.shape[0])
    trail = trail[:min_len]
    trajectory = trajectory[:min_len]
    
    # Compute area between trail and trajectory
    distances = jnp.linalg.norm(trail - trajectory, axis=1)
    area_approx = jnp.sum(distances) * dt
    
    return area_approx

def run_trail_following_experiment():
    """Run a complete trail following experiment."""
    # Parameters
    point_a = jnp.array([0.0, 0.0])
    point_b = jnp.array([0.5, 1.0])
    num_steps = 160
    beta = 10.0  # Trail following strength
    angular_noise_std = 1.0
    sigma = 0.05
    
    # Create a wavy trail
    trail = create_squiggly_line(point_a, point_b, num_points=num_steps+1)
    
    # Simulate agent following the trail
    key = random.PRNGKey(42)
    trajectory = simulate_trail_following(trail, num_steps, beta, angular_noise_std, key, point_a, point_b, sigma)
    
    # Evaluate quality
    quality = evaluate_trail_following_quality(trail, trajectory)
    
    return trail, trajectory, quality

def plot_trail_following_results(trail, trajectory, point_a, point_b, save_path=None):
    """Plot trail following results."""
    plt.figure(figsize=(10, 8))
    
    # Plot trail
    plt.plot(trail[:, 0], trail[:, 1], 'b-', linewidth=2, label='Given Trail', alpha=0.7)
    
    # Plot agent trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=1.5, label='Agent Trajectory')
    
    # Mark start and end points
    plt.scatter(point_a[0], point_a[1], color='green', s=100, label='Start', zorder=5)
    plt.scatter(point_b[0], point_b[1], color='red', s=100, label='Goal', zorder=5)
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Trail Following Experiment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
