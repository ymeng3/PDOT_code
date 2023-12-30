
using CSV, DataFrames, Random, Plots, Distributions, Distances 
include("../src/PDOT.jl")

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


# read from the CSV file 
df = CSV.read("/home/ymeng3/experiments/instance&answer/dataset2_instance_log.csv", DataFrame)
for row in eachrow(df) 

    # extract parameters 
    m = row.m
    n = row.m
    seed1 = row.seed1
    seed2 = row.seed2
    instance_number = row.instance_number
        
    if m != 100
        continue  # Skip this iteration if conditions are not met
    end

    # Print the start of processing for this instance
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
    transport_matrix_filename = joinpath("/home/ymeng3/experiments/instance&answer/dataset2_answer", "final_transport_matrix_instance$(instance_number)_$(m)x$(n).csv")
    CSV.write(transport_matrix_filename, DataFrame(final_transport_matrix, :auto))





    println("Finished processing: m = $m, n = $n, instance_number = $instance_number")


end 
