using CSV, DataFrames, Random, Plots, Distributions, Distances, LinearAlgebra



function logsumexp(X; dims)
    max_X = maximum(X; dims=dims)
    exp_sum = sum(exp.(X .- max_X); dims=dims)
    return max_X + log.(exp_sum)
end

function round_transport_plan(T, a, b)
    # Adjust rows of T to match source distribution 'a'
    row_scaling = min.(a ./ sum(T, dims=2), 1)
    for i in 1:size(T, 1)
        T[i, :] .= T[i, :] .* row_scaling[i]
    end

    # Adjust columns of T to match target distribution 'b'
    col_scaling = min.(b ./ sum(T, dims=1)', 1)
    for j in 1:size(T, 2)
        T[:, j] .= T[:, j] .* col_scaling[j]
    end

    # Compute errors in satisfying the marginal constraints
    err_r = a - vec(sum(T, dims=2))
    err_c = b - vec(sum(T, dims=1)')

    # Normalize row error vector
    err_r_norm = err_r / norm(err_r, 1)

    # Adjust the transport plan based on the normalized error vectors
    T .+= err_r_norm * err_c'

    return T
end

function sinkhorn_log(a, b, M, lambda_reg; tol=1e-4, verbose=false, logging=false, enforce_marginals=false)
    start_time = time()
    time_limit = 2 * 60 * 60  # 2 hours
    
    r, c = size(M)
    G = -M / lambda_reg  # Scaled cost matrix
    log_mu = log.(a)
    log_nu = log.(b)
    u, v = zeros(r), zeros(c)

    log_dict = logging ? Dict("err" => [], "niter" => 0, "converged" => false, "time_elapsed" => 0.0) : nothing
    err = 1.0
    converged = false 
    iter = 0

    while true
        iter += 1 

        # Check if time limit exceeded
        if time() - start_time > time_limit
            println("Time limit of 2 hours exceeded at iteration $iter")
            converged = false
            break
        end

        v = log_nu - vec(logsumexp(G .+ reshape(u, :, 1), dims=1))
        u = log_mu - vec(logsumexp(G .+ reshape(v, 1, :), dims=2))

        T = exp.(G .+ reshape(u, :, 1) .+ reshape(v, 1, :))

        # Error for the target distribution (b)
        err_b = norm(sum(T, dims=1)' .- b, 1) 

        # Error for the source distribution (a)
        err_a = norm(sum(T, dims=2) .- a, 1) 

        err = max(err_a, err_b)

        if logging 
            push!(log_dict["err"], err) 
        end

        if err < tol
            converged = true
            break
        end

        if verbose && iter % 200 == 0
            println("Iteration $iter, Error: $err")
        end
    end
    
    T = exp.(G .+ reshape(u, :, 1) .+ reshape(v, 1, :))

    # Perform rounding step if enabled
    if enforce_marginals
        T = round_transport_plan(T, a, b)
    end

    if logging
        log_dict["niter"] = iter
        log_dict["converged"] = converged
        log_dict["time_elapsed"] = time() - start_time
        return T, log_dict
    else
        return T
    end
end