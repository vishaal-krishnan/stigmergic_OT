"""
Trail Straightening Implementation

This module implements the trail straightening problem: how do agents 
make paths more direct over time through collective behavior?
"""

import jax
import jax.numpy as jnp
from jax import vmap, random, lax
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.fft import fft

def compute_curvature_spline(x_spline, y_spline, t):
    """Compute curvature from spline functions."""
    x_dot = x_spline.derivative()(t)
    y_dot = y_spline.derivative()(t)
    x_ddot = x_spline.derivative(n=2)(t)
    y_ddot = y_spline.derivative(n=2)(t)
    
    numerator = jnp.abs(x_dot * y_ddot - y_dot * x_ddot)
    denominator = (x_dot**2 + y_dot**2)**(3/2)
    
    return numerator / (denominator + 1e-8)

def analyze_trajectory_curvature(trajectory, n_points=1024, smoothing=1e-3):
    """Analyze trajectory curvature and path efficiency."""
    t_raw = np.linspace(0, 1, len(trajectory))
    x_raw, y_raw = trajectory[:, 0], trajectory[:, 1]
    
    # Fit smoothing splines
    x_spline = UnivariateSpline(t_raw, x_raw, s=smoothing)
    y_spline = UnivariateSpline(t_raw, y_raw, s=smoothing)
    
    # Uniform evaluation points
    t_uniform = np.linspace(0, 1, n_points)
    
    # Compute curvature
    curvature = compute_curvature_spline(x_spline, y_spline, t_uniform)
    curvature_detrended = curvature - np.mean(curvature)
    
    # Power spectrum of curvature
    spectrum = np.abs(fft(curvature_detrended))**2
    spectrum = spectrum[:n_points // 2] / len(curvature_detrended)
    
    # Path length ratio (efficiency measure)
    path_length = np.sum(np.sqrt(np.diff(x_raw)**2 + np.diff(y_raw)**2))
    straight_length = np.linalg.norm(trajectory[-1] - trajectory[0])
    path_length_ratio = path_length / straight_length
    
    return spectrum, path_length_ratio, curvature

def create_pheromone_field_from_trajectory(trajectory, X, Y, sigma=0.05):
    """Create pheromone field from trajectory points."""
    field = jnp.zeros_like(X)
    for point in trajectory:
        dx = X - point[0]
        dy = Y - point[1]
        field += jnp.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    return field

def simulate_trail_straightening(initial_trail, num_agents, num_iterations, beta, 
                               angular_noise_std, key, point_a, point_b, sigma=0.05):
    """Simulate multiple agents straightening a trail over iterations."""
    from .trail_following import simulate_trail_following
    
    current_trail = initial_trail
    all_trajectories = []
    
    for iteration in range(num_iterations):
        # Create pheromone field from current trail
        x_vals = jnp.linspace(-0.2, 0.7, 100)
        y_vals = jnp.linspace(-0.2, 1.2, 100)
        X, Y = jnp.meshgrid(x_vals, y_vals)
        pheromone_field = create_pheromone_field_from_trajectory(current_trail, X, Y, sigma)
        
        # Simulate multiple agents
        iteration_trajectories = []
        for agent in range(num_agents):
            key, subkey = random.split(key)
            traj = simulate_trail_following(current_trail, 160, beta, angular_noise_std, 
                                          subkey, point_a, point_b, sigma)
            iteration_trajectories.append(traj)
        
        all_trajectories.append(iteration_trajectories)
        
        # Update trail as average of all trajectories (straightening effect)
        if iteration_trajectories:
            # Find minimum length
            min_len = min(traj.shape[0] for traj in iteration_trajectories)
            # Average all trajectories
            averaged_traj = jnp.mean(jnp.stack([traj[:min_len] for traj in iteration_trajectories]), axis=0)
            current_trail = averaged_traj
    
    return all_trajectories, current_trail

def run_trail_straightening_experiment():
    """Run a complete trail straightening experiment."""
    from .trail_following import create_squiggly_line
    
    # Parameters
    point_a = jnp.array([0.0, 0.0])
    point_b = jnp.array([0.5, 1.0])
    num_agents = 10
    num_iterations = 5
    beta = 5.0
    angular_noise_std = 0.5
    sigma = 0.05
    
    # Create initial wavy trail
    initial_trail = create_squiggly_line(point_a, point_b, amplitude=0.2, frequency=3.0)
    
    # Run straightening simulation
    key = random.PRNGKey(42)
    all_trajectories, final_trail = simulate_trail_straightening(
        initial_trail, num_agents, num_iterations, beta, angular_noise_std, key, point_a, point_b, sigma
    )
    
    # Analyze straightening
    initial_spectrum, initial_ratio, _ = analyze_trajectory_curvature(initial_trail)
    final_spectrum, final_ratio, _ = analyze_trajectory_curvature(final_trail)
    
    return {
        'initial_trail': initial_trail,
        'final_trail': final_trail,
        'all_trajectories': all_trajectories,
        'initial_efficiency': initial_ratio,
        'final_efficiency': final_ratio,
        'initial_spectrum': initial_spectrum,
        'final_spectrum': final_spectrum
    }

def plot_trail_straightening_results(results, save_path=None):
    """Plot trail straightening results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Initial vs Final Trail
    ax1 = axes[0, 0]
    ax1.plot(results['initial_trail'][:, 0], results['initial_trail'][:, 1], 
             'b-', linewidth=2, label='Initial Trail', alpha=0.7)
    ax1.plot(results['final_trail'][:, 0], results['final_trail'][:, 1], 
             'r-', linewidth=2, label='Final Trail')
    ax1.set_title('Trail Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Efficiency over iterations
    ax2 = axes[0, 1]
    iterations = range(len(results['all_trajectories']))
    efficiencies = []
    for i, trajs in enumerate(results['all_trajectories']):
        if trajs:
            min_len = min(traj.shape[0] for traj in trajs)
            avg_traj = jnp.mean(jnp.stack([traj[:min_len] for traj in trajs]), axis=0)
            _, ratio, _ = analyze_trajectory_curvature(avg_traj)
            efficiencies.append(ratio)
    
    ax2.plot(iterations, efficiencies, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Path Efficiency (lower is better)')
    ax2.set_title('Convergence to Straight Path')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Curvature spectrum comparison
    ax3 = axes[1, 0]
    freqs = np.linspace(0, 0.5, len(results['initial_spectrum']))
    ax3.loglog(freqs[1:], results['initial_spectrum'][1:], 'b-', label='Initial', alpha=0.7)
    ax3.loglog(freqs[1:], results['final_spectrum'][1:], 'r-', label='Final')
    ax3.set_xlabel('Frequency')
    ax3.set_ylabel('Power Spectrum')
    ax3.set_title('Curvature Spectrum')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Sample trajectories from last iteration
    ax4 = axes[1, 1]
    if results['all_trajectories']:
        last_trajs = results['all_trajectories'][-1]
        for i, traj in enumerate(last_trajs[:5]):  # Show first 5 trajectories
            ax4.plot(traj[:, 0], traj[:, 1], alpha=0.6, linewidth=1)
        ax4.plot(results['final_trail'][:, 0], results['final_trail'][:, 1], 
                 'k-', linewidth=3, label='Averaged Trail')
        ax4.set_title('Sample Trajectories (Final Iteration)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
