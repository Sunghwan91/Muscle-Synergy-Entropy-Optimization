# Entropy-Based Post-Processing for Robust Muscle Synergy Extraction

This repository contains the Julia implementation and a sample anonymized electromyography (EMG) dataset for the paper: **"Entropy-Based Post-Processing for Robust Muscle Synergy Extraction via NMF"** (Submitted to IEEE TNSRE).

## 📌 Overview
Non-negative Matrix Factorization (NMF) is widely used for muscle synergy extraction, but it suffers from rotational ambiguity, leading to non-unique solutions. This repository provides a post-processing optimization framework that selects a stable, canonical representative synergy set from the feasible equivalence class by minimizing the Shannon entropy of the spatial synergy vectors. 

Crucially, this method preserves the original Variance Accounted For (VAF) and improves spatial sparsity and functional distinctness.

## ⚙️ Prerequisites
To run this code, you need [Julia](https://julialang.org/) installed along with the following packages:
- `JuMP` (Optimization modeling framework)
- `Ipopt` (Nonlinear optimization solver)
- `LinearAlgebra` (Standard linear algebra operations)
- `Zygote` (Algorithmic differentiation for gradients)

You can install these packages in the Julia REPL by typing `]` to enter the Pkg prompt and running:
```julia
pkg> add JuMP Ipopt LinearAlgebra Zygote
