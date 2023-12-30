
using CSV, DataFrames, Random, Plots, Distributions, Distances 
include("../src/PDOT.jl")

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




# read from the CSV file 
df = CSV.read("/home/ymeng3/experiments/instance&answer/dataset1_instance_log.csv", DataFrame)
for row in eachrow(df) 

    # extract parameters 
    m = row.m
    n = row.n
    seed1 = row.seed1
    seed2 = row.seed2
    instance_number = row.instance_number
        
    if m != 10000 || !(5 <= instance_number <= 10)
        continue  # Skip this iteration if conditions are not met
    end
    
    # Print the start of processing for this instance
    println("Starting processing: m = $m, n = $n, instance_number = $instance_number")


    # genereate the instance 
    x_s, x_t, cost_matrix = generate_instance(m, n, seed1, seed2)


    # Normalize the cost matrix and set up the problem
    min_val = minimum(cost_matrix)
    max_val = maximum(cost_matrix)
    normalized_cost_matrix = (cost_matrix .- min_val) ./ (max_val - min_val)
    p = ones(m)./m
    q = ones(n)./n
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
    # CSV.write("/Users/justinmeng/Desktop/Project OR2/synthetic_experiment_dataset1_v2/results_$(m)instance$(instance_number).csv", results_df)

    # # Prepare the data for plotting
    # x_values = (evaluation_frequency:evaluation_frequency:iter)  # Iterations where KKT residuals were recorded
    # y_values = kkt_stats_res  # The recorded KKT residuals

    # # Plot the data
    # kkt_plt = Plots.plot(x_values, y_values, xlabel="Iterations", ylabel="KKT Residual", yscale=:log10, label=false)

    # # Save the plot
    # plot_filename = "/Users/justinmeng/Desktop/Project OR2/synthetic_experiment_dataset1_v2/plot_$(m)instance$(instance_number).png"
    # Plots.savefig(kkt_plt, plot_filename)



    
    # Save the final transport matrix for each instance
    transport_matrix_filename = joinpath("/home/ymeng3/experiments/instance&answer/dataset1_answer", "final_transport_matrix_instance$(instance_number)_$(m)x$(n).csv")
    CSV.write(transport_matrix_filename, DataFrame(final_transport_matrix, :auto))



    

    # Print the end of processing for this instance
    println("Finished processing: m = $m, n = $n, instance_number = $instance_number")


end 



