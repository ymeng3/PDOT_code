
using CSV, DataFrames, Random, Plots, Distributions, Distances, Images, MLDatasets
include("../src/solver.jl")


function load_mnist_data()
    train_x, train_y = MNIST.traindata()
    return train_x, train_y
end

train_x, train_y = load_mnist_data()
images = train_x  # train_x is already a 28x28x60000 array
total_images = size(images)[3]  # 60000


function pre(mnistex::Array{Float64,2})
    temp = copy(mnistex)
    temp[temp .== 0] .= 1e-6
    temp /= sum(temp)
    return reshape(temp, :, 1)
end

function process_image(image::Array{UInt8, 2}, size::Int)
    # Convert the image to a format that can be used with the Images package
    img = Gray.(reshape(image, 28, 28))
    
    # Resize the image
    resized_image = imresize(img, (size, size))

    # Convert the resized image to an array and preprocess
    processed_image = Array{Float64}(resized_image)
    return pre(processed_image)
end

function process_image(image::Matrix{N0f8}, size::Int)
    # Resize the image
    resized_image = imresize(image, (size, size))

    # Convert the resized image to an array of Float64 and preprocess
    processed_image = Float64.(resized_image)
    return pre(processed_image)
end





function cartesian_product(arrays...)
    n = length(arrays)
    result = [vec(a) for a in arrays]
    for i in 1:n
        result[i] = repeat(result[i], inner=prod(length.(arrays)[i+1:end]), outer=prod(length.(arrays)[1:i-1]))
    end
    return hcat(result...)
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
df = CSV.read("/Users/justinmeng/Desktop/Project OR2/experiments/instance&answer/dataset3_instance_log.csv", DataFrame)
for row in eachrow(df) 

    # extract parameters 
    m = row.m
    n = row.m
    seed1 = row.seed1
    seed2 = row.seed2
    instance_number = row.instance_number
        
    println("#####################################")
    # Print the start of processing for this instance
    println("Starting processing: m = $m, n = $n, instance_number = $instance_number")


    # Create separate random number generators for p and q
    rng1 = MersenneTwister(seed1)
    rng2 = MersenneTwister(seed2)

    # Generate random indices using separate RNGs
    p_index = rand(rng1, 1:total_images)
    q_index = rand(rng2, 1:total_images)

    # Access the selected images
    p_image = images[:, :, p_index]
    q_image = images[:, :, q_index]

    # Process the images
    p = process_image(p_image, m)
    q = process_image(q_image, m)

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


    lambda_reg = 0.1
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
    transport_matrix_filename = joinpath("/Users/justinmeng/Desktop/Project OR2/experiments/result/sinkhorn0.1_dataset3_res", "transport_matrix_instance$(instance_number)_$(m)x$(n).csv")
    CSV.write(transport_matrix_filename, DataFrame(final_transport_matrix, :auto))


    println("Finished processing: m = $m, n = $n, instance_number = $instance_number")
    println("#####################################")
end

log_filename = joinpath("/Users/justinmeng/Desktop/Project OR2/experiments/result/sinkhorn0.1_dataset3_res", "sinkhorn0.1_dataset3_log.csv") 
CSV.write(log_filename, log_df)


