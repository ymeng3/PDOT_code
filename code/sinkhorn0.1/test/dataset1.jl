using CSV, DataFrames, Random, Plots, Distributions, Distances, LinearAlgebra
include("../src/solver.jl")


# generate instance from seed in csv file
function generate_instance(m, n, seed1, seed2)
    d = 2 

    # generate x_s 
    Random.seed!(seed1) 
    rng = Random.MersenneTwister(seed1) 
    mu_s = rand(rng, Normal(0.0, 1.0), d)
    A_s = rand(rng, Normal(0.01, 1.0), (d, d))
    cov_s = A_s * A_s'
    x_s = rand(rng, MvNormal(mu_s, cov_s), m) 

    # generate x_t
    Random.seed!(seed2)
    rng = Random.MersenneTwister(seed2)
    mu_t = rand(rng, Normal(0.0, 1.0), d)
    A_t = rand(rng, Normal(0.01, 1.0), (d, d))
    cov_t = A_t * A_t'
    x_t = rand(rng, MvNormal(mu_t, cov_t), n)

    # compute cost matrix 
    cost_matrix = pairwise(Euclidean(), x_s, x_t, dims=2).^2

    return x_s, x_t, cost_matrix
end 


# log dataframe 
log_df = DataFrame(
    m = Int[],
    n = Int[],
    instance_number = Int[],
    primal_infeasibility = Float64[],
    time_full = Float64[],
    converged = Bool[],
    num_iterations = Int[]
)

# read from the CSV file                                                  
df = CSV.read("/Users/justinmeng/Desktop/Project OR2/experiments/instance&answer/dataset1_instance_log.csv", DataFrame)  
for row in eachrow(df) 

    # extract parameters 
    m = row.m
    n = row.n
    seed1 = row.seed1
    seed2 = row.seed2
    instance_number = row.instance_number

    
    # Print the start of processing for this instance
    println("#####################################")
    println("Starting processing: m = $m, n = $n, instance_number = $instance_number")


    # genereate the instance 
    x_s, x_t, cost_matrix = generate_instance(m, n, seed1, seed2)


    # Normalize the cost matrix and set up the problem
    min_val = minimum(cost_matrix) 
    max_val = maximum(cost_matrix) 
    normalized_cost_matrix = (cost_matrix .- min_val) ./ (max_val - min_val) 
    p = ones(m)./m 
    q = ones(n)./n 

    lambda_reg = 0.1
    final_transport_matrix, log_dict = sinkhorn_knopp(p, q, normalized_cost_matrix, lambda_reg; tol=1e-4, verbose=false, log=true, enforce_marginals=true)

    println("Instance $instance_number: Lambda_reg = $lambda_reg")
    println("Number of iterations: ", log_dict["niter"])
    println("Time taken: ", log_dict["time_elapsed"], " seconds")
    println("Converged: ", log_dict["converged"])


    sum_rows = sum(final_transport_matrix, dims=2)  # Sum over columns
    sum_cols = sum(final_transport_matrix, dims=1)  # Sum over rows
    source_diff = maximum(abs.(sum_rows[:] .- p))  # Max absolute difference from source distribution
    target_diff = maximum(abs.(sum_cols[:] .- q))  # Max absolute difference from target distribution
    primal_infeasibility = max(source_diff, target_diff)
    println("Primal Infeasibility for instance $instance_number: $primal_infeasibility")


    # append to log dataframe
    push!(log_df, (m=m, n=n, instance_number=instance_number, primal_infeasibility=primal_infeasibility, time_full=log_dict["time_elapsed"], converged=log_dict["converged"], num_iterations=log_dict["niter"]))


    # result dataframe 
    # Save the final transport matrix for each instance
    transport_matrix_filename = joinpath("/Users/justinmeng/Desktop/Project OR2/experiments/result/sinkhorn0.1_dataset1_res ", "transport_matrix_instance$(instance_number)_$(m)x$(n).csv")
    CSV.write(transport_matrix_filename, DataFrame(final_transport_matrix, :auto))


    # Print the end of processing for this instance
    println("Finished processing: m = $m, n = $n, instance_number = $instance_number")
    println("#####################################")

end 


log_filename = joinpath("/Users/justinmeng/Desktop/Project OR2/experiments/result/sinkhorn0.1_dataset1_res ", "sinkhorn0.1_dataset1_log.csv") 
CSV.write(log_filename, log_df)
