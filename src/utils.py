"""
Utility functions for stigmergic optimal transport experiments.
"""

from .trail_following import run_trail_following_experiment
from .trail_straightening import run_trail_straightening_experiment
from .inhomogeneous_optimization import run_inhomogeneous_optimization_experiment

def run_all_experiments():
    """Run all three problem experiments and return results."""
    print("=" * 60)
    print("Running Stigmergic Optimal Transport Experiments")
    print("=" * 60)
    
    print("\n[1/3] Running Trail Following Experiment...")
    trail, trajectory, quality = run_trail_following_experiment()
    print(f"      Trail following quality (lower is better): {quality:.4f}")
    
    print("\n[2/3] Running Trail Straightening Experiment...")
    straightening_results = run_trail_straightening_experiment()
    print(f"      Initial efficiency: {straightening_results['initial_efficiency']:.4f}")
    print(f"      Final efficiency: {straightening_results['final_efficiency']:.4f}")
    print(f"      Improvement: {(1 - straightening_results['final_efficiency']/straightening_results['initial_efficiency'])*100:.1f}%")
    
    print("\n[3/3] Running Inhomogeneous Media Optimization...")
    optimization_results = run_inhomogeneous_optimization_experiment()
    print(f"      Optical path length: {optimization_results['optical_length']:.4f}")
    print(f"      Snell optimal: {optimization_results['snell_optimal']:.4f}")
    print(f"      Efficiency: {optimization_results['efficiency']:.2%}")
    
    print("\n" + "=" * 60)
    print("All experiments completed successfully!")
    print("=" * 60)
    
    return {
        'trail_following': (trail, trajectory, quality),
        'trail_straightening': straightening_results,
        'inhomogeneous_optimization': optimization_results
    }

