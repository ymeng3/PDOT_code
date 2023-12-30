using CSV, DataFrames, Random, Plots, Distributions, Distances, LinearAlgebra
include("../src/solver.jl")


# Test the function
Random.seed!(123)
m, n = 100, 100
C = rand(m, n) * 10
p = abs.(randn(m)); p /= sum(p)
q = abs.(randn(n)); q /= sum(q)

lambda_reg_values = [0.01, 0.1, 1.0]
for lambda_reg in lambda_reg_values
    println("Lambda_reg: $lambda_reg")
    final_transport_matrix = sinkhorn_log(p, q, C, lambda_reg, max_iter=20000, rounding=true)
    

    # log dataframe 
    sum_rows = sum(final_transport_matrix, dims=2)  # Sum over columns
    sum_cols = sum(final_transport_matrix, dims=1)  # Sum over rows
    source_diff = maximum(abs.(sum_rows[:] .- p))  # Max absolute difference from source distribution
    target_diff = maximum(abs.(sum_cols[:] .- q))  # Max absolute difference from target distribution
    primal_infeasibility = max(source_diff, target_diff)
    println("Primal Infeasibility: $primal_infeasibility")


end
