println("Starting dataset3.jl script...")

println("Julia version: ", VERSION)
println("Current working directory: ", pwd())
println("Loaded packages: ")
using Pkg
Pkg.status()


using CSV, DataFrames, Random, Distributions, Distances, Images, MLDatasets
include("../src/PDOT.jl")


# function load_mnist_data()
#     train_x, train_y = MNIST.traindata()
#     return train_x, train_y
# end 

# train_x, train_y = load_mnist_data()

train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()
println("MNIST dataset loaded successfully.")

images = train_x  # train_x is already a 28x28x60000 array
total_images = size(images)[3]  # 60000


function pre(mnistex::Array{Float64,2})
    temp = copy(mnistex)
    temp[temp .== 0] .= 1e-6
    temp /= sum(temp)
    return reshape(temp, :, 1)
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



# read from the CSV file 
df = CSV.read("/home/ymeng3/experiments/instance&answer/dataset3_instance_log.csv", DataFrame)
for row in eachrow(df) 

    # extract parameters 
    m = row.m
    n = row.m
    seed1 = row.seed1
    seed2 = row.seed2
    instance_number = row.instance_number


    # if !(m in [10, 17, 32])
    #     continue  # Skip this iteration if m is not in the specified values
    # end

    # if m != 55
    #     continue  # Skip this iteration if conditions are not met
    # end
    
    # if m != 100
    #     continue  # Skip this iteration if conditions are not met
    # end
        
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


    problem = OptimalTransportProblem(normalized_cost_matrix, p, q)

    # parameters for optimization 
    params = PrimalDualOptimizerParameters(
        2 * 60 * 60,  # 2 hours
        1e-6,
        ConstantStepsizeParams(),
    )



    # Run the optimization
    kkt_stats_res, iter, time_basic, time_full, converged, final_transport_matrix = optimize(problem, params)

    # # Initialize an array for KKT residuals with NaN and populate it every 64 iterations
    # kkt_res_with_nans = fill(NaN, iter)
    # evaluation_frequency = 64  
    # for i in 1:length(kkt_stats_res)
    #     kkt_res_with_nans[evaluation_frequency * i] = kkt_stats_res[i]
    # end

    # # Create a DataFrame
    # results_df = DataFrame(
    #     Iteration = 1:iter,
    #     KKT_Residual = kkt_res_with_nans,
    #     Time_Basic = repeat([time_basic], iter),
    #     Time_Full = repeat([time_full], iter),
    #     Converged = repeat([converged], iter),
    # )

    # # Save DataFrame to a CSV file
    # CSV.write("/Users/justinmeng/Desktop/Project OR2/experiments/pdlp_dataset2_time1/results_$(m)instance$(instance_number).csv", results_df)

    # # Prepare the data for plotting
    # x_values = (evaluation_frequency:evaluation_frequency:iter)  # Iterations where KKT residuals were recorded
    # y_values = kkt_stats_res  # The recorded KKT residuals

    # # Plot the data
    # kkt_plt = Plots.plot(x_values, y_values, xlabel="Iterations", ylabel="KKT Residual", yscale=:log10, label=false)

    # # Save the plot
    # plot_filename = "/Users/justinmeng/Desktop/Project OR2/experiments/pdlp_dataset2_time1/plot_$(m)instance$(instance_number).png"
    # Plots.savefig(kkt_plt, plot_filename)


    # Save the final transport matrix for each instance
    transport_matrix_filename = joinpath("/home/ymeng3/experiments/instance&answer/dataset3_answer", "final_transport_matrix_instance$(instance_number)_$(m)x$(n).csv")
    CSV.write(transport_matrix_filename, DataFrame(final_transport_matrix, :auto))





    println("Finished processing: m = $m, n = $n, instance_number = $instance_number")


end 

