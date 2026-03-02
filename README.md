# Entropy-Based Post-Processing for Robust Muscle Synergy Extraction

This repository contains the Julia implementation and a sample anonymized electromyography (EMG) dataset for the paper: **Entropy-Based Post-Processing for Robust Muscle Synergy Extraction: Selecting a Canonical Representative from NMF Solutions**.

## Overview
Non-negative Matrix Factorization (NMF) is widely used for muscle synergy extraction, but it suffers from rotational ambiguity, leading to non-unique solutions. This repository provides a post-processing optimization framework that selects a stable, canonical representative synergy set from the feasible equivalence class by minimizing the Shannon entropy of the spatial synergy vectors. 

## Prerequisites
To run this code, you need [Julia](https://julialang.org/) installed along with the following packages:
- `JuMP` (Optimization modeling framework)
- `Ipopt` (Nonlinear optimization solver)
- `LinearAlgebra` (Standard linear algebra operations)
- `Zygote` (Algorithmic differentiation for gradients)

You can install these packages in the Julia REPL by typing `]` to enter the Pkg prompt and running:
```julia
pkg> add JuMP Ipopt LinearAlgebra Zygote
```

## Repository Structure
entropy_optimization.jl: The core function new_entropy_optimize(C, W; lambda) that performs the entropy minimization using IPOPT.

sample_data/: Contains an anonymized sample gait EMG dataset (Temporal activations C and spatial synergies W derived from initial NMF).

## How to Use
Clone the repository or download the files.

Open your Julia REPL and include the script.

Pass your temporal activation matrix (C) and spatial synergy matrix (W) into the function.

```Julia
# Include the optimization function
include("entropy_optimization.jl")

# Load your initial NMF results (Example dimensions)
# C_init: Temporal activations (Time x Synergies, e.g., 101 x 4)
# W_init: Spatial synergies (Synergies x Muscles, e.g., 4 x 8)
# (Replace these with your actual loaded data matrices)

# Run the optimization
# lambda controls the penalty for slack variables (default is 1e1)
optimized_transformation_matrix = new_entropy_optimize(C_init, W_init; lambda = 10.0)

# Calculate the newly optimized spatial synergies
W_new = optimized_transformation_matrix * W_init
