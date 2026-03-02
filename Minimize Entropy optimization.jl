using JuMP
using Ipopt
using LinearAlgebra
using Zygote

function Entropy_optimize(C::Matrix, W::Matrix; lambda = 1e1) # Simple entropy optimization
    # C: Temporal activations matrix (Time x Synergies, e.g., 101 x r)
    # W: Spatial synergies matrix (Synergies x Muscles, e.g., r x n)

    # Initialize dimensions and matrices
    r, n = size(W)
    A0 = Matrix{Float64}(I, r, r)   # Standard identity matrix for transformation
    S0 = zeros(size(C, 1), r)       # Initialize slack variables with zeros

    # Set up the Ipopt optimization model
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 2000)
    set_optimizer_attribute(model, "print_level", 0)
    
    # Define decision variables
    @variable(model, A[i=1:r, j=1:r], start = A0[i,j])       # Transformation matrix
    @variable(model, A_[1:r, 1:r])                           # Inverse of transformation matrix
    @variable(model, S[i=1:size(C,1), j=1:r], start = S0[i,j]) # Slack variables

    # ==========================================================
    # 1. Calculate baseline condition number of the initial spatial synergy (W)
    # svdvals returns singular values in descending order.
    sigma_min_W = svdvals(W)[1] / svdvals(W)[end]
    target_sigma_min = min(0.8 * sigma_min_W, 1000.0)

    # 2. Define the function for the condition number of the transformed matrix (A * W)
    min_sv_AW_func(a...) = svdvals(reshape([a...], (r, r)) * W)[1] / svdvals(reshape([a...], (r, r)) * W)[end]
    
    # 3. Define the gradient function using Zygote
    function ∇min_sv_AW_func(g::AbstractVector{<:Real}, a::Real...)
        # Zygote.gradient returns a tuple, so we iterate and assign
        grad_tuple = Zygote.gradient(min_sv_AW_func, a...)
        for i in 1:length(a)
            g[i] = grad_tuple[i]
        end
        return
    end

    # 4. Register the custom function and its gradient to the JuMP model
    register(model, :min_sv_AW, r*r, min_sv_AW_func, ∇min_sv_AW_func; autodiff=false)

    # 5. Add a nonlinear constraint for the condition number upper bound
    @NLconstraint(model, min_sv_AW(A...) <= target_sigma_min)


    # ==========================================================
    # Add structural and non-negativity constraints
    @constraint(model, A_ * A .== Matrix{Float64}(I, r, r))  # A_ must be the inverse of A
    @constraint(model, [i=1:r, j=1:n], (A*W)[i,j] >= 0)      # Non-negativity for new spatial synergies
    @constraint(model, [i=1:size(C,1), j=1:r], (C*A_ + S)[i,j] >= 0) # Relaxed non-negativity for temporal activations
    @constraint(model, [i=1:size(C,1), j=1:r], S[i,j] >= 0)  # Slack variables must be non-negative
    @constraint(model, [i=1:r], sum(A[i,:]) == 1.0)          # The sum of each row in matrix A must equal 1.0

    # Define nonlinear expressions for Entropy calculation
    @NLexpression(model, V[i=1:r, j=1:n], sum(A[i,k] * W[k,j] for k in 1:r)) # Transformed spatial synergies (V = A * W)
    @NLexpression(model, ColSums[j=1:n], sum(V[i,j] for i in 1:r))           # (Not used in objective, but kept for reference)
    @NLexpression(model, RowSums[i=1:r], sum(V[i,j] for j in 1:n))           # Sum of values for each synergy vector
    eps_val = 1e-6   

    # (Optional: Norm calculations, can be removed if strictly not used in objective)
    @NLexpression(model, L1_Norms[i=1:r], sum(V[i,j] for j in 1:n)) 
    @NLexpression(model, L2_Norms[i=1:r], sqrt(sum(V[i,j]^2 for j in 1:n) + 1e-8))

    @NLobjective(model, Min,         
        # Minimize the Shannon Entropy of the transformed spatial synergy row vectors
        # Note: Fixed the potential BoundsError by using RowSums[i] inside the log2 function
        -sum(
            (max(V[i,j], eps_val) / max(RowSums[i], eps_val)) * log2(max(V[i,j], eps_val) / max(RowSums[i], eps_val))
            for j in 1:n, i in 1:r
        ) + sum(S[i,j] for i in 1:size(C,1), j in 1:r) * lambda # Penalty term for slack variables
    )

    # Execute the optimization process
    optimize!(model)
    A_optimized = value.(A)

    return A_optimized
end
