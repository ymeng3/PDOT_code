using CSV, DataFrames, Random, Plots, Distributions, Distances, LinearAlgebra
include("../src/solver.jl")


function synthetic_img_input(m, fraction_fg, seed)
    Random.seed!(seed)

    fg_max_intensity = 10
    bg_max_intensity = 1

    IMG = bg_max_intensity * rand(m, m)

    img_i = rand(1:m)
    img_j = rand(1:m)

    length_fg_side = floor(Int, m * sqrt(fraction_fg))

    for i in 0:length_fg_side-1
        for j in 0:length_fg_side-1
            ind_i = mod(img_i + i, m)
            ind_j = mod(img_j + j, m)
            IMG[ind_i == 0 ? m : ind_i, ind_j == 0 ? m : ind_j] = fg_max_intensity * rand()
        end
    end
    return IMG
end

function presyn(synthetic)
    synthetic ./= sum(synthetic)
    return reshape(synthetic, :, 1)
end

function cartesian_product(arrays...)
    n = length(arrays)
    result = [vec(a) for a in arrays]
    for i in 1:n
        result[i] = repeat(result[i], inner=prod(length.(arrays)[i+1:end]), outer=prod(length.(arrays)[1:i-1]))
    end
    return hcat(result...)
end

fraction_fg = 0.2 


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
df = CSV.read("/Users/justinmeng/Desktop/Project OR2/experiments/instance&answer/dataset2_instance_log.csv", DataFrame)
for row in eachrow(df) 

    # extract parameters 
    m = row.m
    n = row.m
    seed1 = row.seed1
    seed2 = row.seed2
    instance_number = row.instance_number
        
    # Print the start of processing for this instance
    println("#####################################")
    println("Starting processing: m = $m, n = $n, instance_number = $instance_number")

    p = presyn(synthetic_img_input(m, fraction_fg, seed1))
    q = presyn(synthetic_img_input(m, fraction_fg, seed2))
    
    # Flatten p and q to have shape (m^2, )
    p = p[:]
    q = q[:]
    
    # Generate and normalize the cost matrix C
    C = 0:m-1
    C = collect(C)  # Convert UnitRange to Vector
    C = cartesian_product(C, C)    
    C = pairwise(Minkowski(1), C', dims=2)
    C ./= maximum(C)  # Normalize C so that its maximum value is 1
    
    # Further normalize C to have values between 0 and 1
    min_val = minimum(C)
    max_val = maximum(C)
    normalized_cost_matrix = (C .- min_val) ./ (max_val - min_val)

    lambda_reg = 0.01
    final_transport_matrix, log_dict = sinkhorn_knopp(p, q, normalized_cost_matrix, lambda_reg; tol=1e-4, verbose=true, log=true, enforce_marginals=true)

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
    transport_matrix_filename = joinpath("/Users/justinmeng/Desktop/Project OR2/experiments/result/sinkhorn0.01_dataset2_res", "transport_matrix_instance$(instance_number)_$(m)x$(n).csv")
    CSV.write(transport_matrix_filename, DataFrame(final_transport_matrix, :auto))


    println("Finished processing: m = $m, n = $n, instance_number = $instance_number")
    println("#####################################")
end

log_filename = joinpath("/Users/justinmeng/Desktop/Project OR2/experiments/result/sinkhorn0.01_dataset2_res", "sinkhorn0.01_dataset2_log.csv") 
CSV.write(log_filename, log_df)
