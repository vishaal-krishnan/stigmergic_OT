"""
Stigmergic Optimal Transport Package

A computational framework for studying collective path optimization through 
stigmergic feedback in inhomogeneous media.
"""

from .trail_following import (
    run_trail_following_experiment,
    simulate_trail_following,
    evaluate_trail_following_quality
)

from .trail_straightening import (
    run_trail_straightening_experiment,
    simulate_trail_straightening,
    analyze_trajectory_curvature
)

from .inhomogeneous_optimization import (
    run_inhomogeneous_optimization_experiment,
    simulate_inhomogeneous_optimization,
    compute_optical_path_length,
    snell_optimal_path_length
)

from .utils import run_all_experiments

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # Trail following
    "run_trail_following_experiment",
    "simulate_trail_following", 
    "evaluate_trail_following_quality",
    
    # Trail straightening
    "run_trail_straightening_experiment",
    "simulate_trail_straightening",
    "analyze_trajectory_curvature",
    
    # Inhomogeneous optimization
    "run_inhomogeneous_optimization_experiment",
    "simulate_inhomogeneous_optimization",
    "compute_optical_path_length",
    "snell_optimal_path_length",
    
    # Utilities
    "run_all_experiments"
]
