using Random, LinearAlgebra
using OptimalTransport
include("../src/solver.jl")



# Test the function
Random.seed!(123)
m, n = 100, 100
# Create cost matrix and distributions
C = rand(m, n) * 10  # Cost matrix
p = abs.(randn(m))   # Source distribution
q = abs.(randn(n))   # Target distribution

# Normalize distributions
p /= sum(p)
q /= sum(q)

# Range of lambda_reg values to try
lambda_reg_values = [0.01, 0.1, 1.0]
for lambda_reg in lambda_reg_values
    T, log_dict = sinkhorn_knopp(p, q, C, lambda_reg; log=true, enforce_marginals=true)
    println("Lambda_reg: $lambda_reg, Number of iterations: ", length(log_dict["err"]))
end

