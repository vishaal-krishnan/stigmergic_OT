# Stigmergic Optimal Transport

**Computational framework for studying collective path optimization through stigmergic feedback in inhomogeneous media.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This repository contains implementations for three fundamental problems in stigmergic optimal transport:

1. **Trail Following** - How agents follow existing pheromone trails
2. **Trail Straightening** - How collective behavior leads to path optimization  
3. **Inhomogeneous Media Optimization** - How agents optimize paths through refractive environments

## Repository Structure

```
stigmergic_OT/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── LICENSE               # MIT License
│
├── src/                  # Core implementations
│   ├── __init__.py
│   ├── trail_following.py
│   ├── trail_straightening.py
│   ├── inhomogeneous_optimization.py
│   └── utils.py
│
└── notebooks/            # Interactive demonstrations
    ├── 01_trail_following.ipynb
    ├── 02_trail_straightening.ipynb
    └── 03_inhomogeneous_optimization.ipynb
```

## Installation

```bash
git clone https://github.com/yourusername/stigmergic-optimal-transport.git
cd stigmergic-optimal-transport
pip install -r requirements.txt
```

### Requirements

- JAX >= 0.4.0 (for high-performance computing)
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- SciPy >= 1.8.0
- Jupyter >= 1.0.0

## Quick Start

### Using Python Scripts

```python
from src import run_all_experiments

# Run all three experiments
results = run_all_experiments()

# Access individual results
trail_following = results['trail_following']
straightening = results['trail_straightening']
optimization = results['inhomogeneous_optimization']
```

### Using Jupyter Notebooks

For interactive exploration with visualizations:

```bash
jupyter notebook notebooks/
```

Then open:
- `01_trail_following.ipynb` - Agent navigation along pheromone trails
- `02_trail_straightening.ipynb` - Collective path optimization
- `03_inhomogeneous_optimization.ipynb` - Refractive path finding

## Problem Descriptions

### 1. Trail Following

**Question:** Given an existing pheromone trail, how do agents follow it?

**Method:** Agents sense pheromone gradients and adjust heading perpendicular to their current direction.

**Key Equation:**
```
dθ/dt = β ∇φ · n̂ + √(2Dθ) η(t)
```

### 2. Trail Straightening

**Question:** How do multiple agents collectively make paths more direct?

**Method:** Iterative process where agents follow trails, average their paths, and the result becomes the new trail.

**Key Insight:** Collective averaging naturally reduces curvature fluctuations.

### 3. Inhomogeneous Media Optimization

**Question:** How do agents find optimal paths through media with varying refractive indices?

**Method:** Agents respond to both pheromone gradients and refractive index gradients, discovering paths that minimize optical length.

**Key Result:** Emergent behavior satisfies Snell's law at interfaces.

## Theory Background

This work combines concepts from:

- **Optimal Transport Theory** - Mathematical framework for minimizing transport costs
- **Stigmergy** - Indirect coordination through environmental modification
- **Calculus of Variations** - Finding paths that minimize functionals
- **Statistical Physics** - Stochastic processes with noise and drift

### Physical Interpretation

The stigmergic approach provides a **physically realizable** mechanism for solving optimal transport problems, unlike abstract optimization algorithms. Agents use only:
- Local gradient information
- Noisy sensing
- Simple movement rules

Yet they collectively solve global optimization problems!

## Usage Examples

### Trail Following

```python
from src.trail_following import (
    run_trail_following_experiment,
    create_squiggly_line,
    simulate_trail_following
)
import jax.numpy as jnp

# Create a test trail
point_a = jnp.array([0.0, 0.0])
point_b = jnp.array([0.5, 1.0])
trail = create_squiggly_line(point_a, point_b)

# Simulate agent following it
trail, trajectory, quality = run_trail_following_experiment()
print(f"Trail following quality: {quality:.4f}")
```

### Trail Straightening

```python
from src.trail_straightening import run_trail_straightening_experiment

results = run_trail_straightening_experiment()
print(f"Initial efficiency: {results['initial_efficiency']:.4f}")
print(f"Final efficiency: {results['final_efficiency']:.4f}")
```

### Inhomogeneous Optimization

```python
from src.inhomogeneous_optimization import run_inhomogeneous_optimization_experiment

results = run_inhomogeneous_optimization_experiment()
print(f"Efficiency vs Snell's law: {results['efficiency']:.2%}")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.