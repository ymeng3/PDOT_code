using CSV, DataFrames, Random, Plots, Distributions, Distances, LinearAlgebra

function sinkhorn_knopp(a, b, M, reg; tol=1e-4, verbose=false, log=false, enforce_marginals=false)
    start_time = time()
    time_limit = 2 * 60 * 60  # 2 hours

    r, c = size(M)
    K = exp.(-M / reg)

    u = ones(r) / r
    v = ones(c) / c

    log_dict = log ? Dict("err" => [], "niter" => 0, "converged" => false, "time_elapsed" => 0.0) : nothing
    err = 1.0
    converged = false 
    iter = 0

    while true  # Remove the max_iter constraint
        iter += 1

        # Check if time limit exceeded
        if time() - start_time > time_limit
            println("Time limit of 2 hours exceeded at iteration $iter")
            converged = false
            break
        end

        u_prev, v_prev = u, v
        u .= a ./ (K * v)
        v .= b ./ (K' * u)

        tmp2 = sum((Diagonal(u) * K * Diagonal(v)), dims=1)
        err_b = norm(tmp2' - b, 1)  

        tmp1 = sum((Diagonal(u) * K * Diagonal(v)), dims=2)
        err_a = norm(tmp1 - a, 1) 

        err = max(err_a, err_b) 
        
        if log 
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

    T = Diagonal(u) * K * Diagonal(v)

    if enforce_marginals
        # Adjust T to satisfy marginal constraints
        T = round_transport_plan(T, a, b)
    end

    if log
        log_dict["niter"] = iter
        log_dict["converged"] = converged
        log_dict["u"] = u
        log_dict["v"] = v
        log_dict["T"] = T
        log_dict["time_elapsed"] = time() - start_time
        return T, log_dict
    else
        return T
    end
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



