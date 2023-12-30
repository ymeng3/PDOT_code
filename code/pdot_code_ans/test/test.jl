import Random 
import Plots
include("../src/PDOT.jl")


ran_seed = 123
Random.seed!(ran_seed)
rng = Random.MersenneTwister(ran_seed)

m, n = 100, 100

C = rand(rng, m, n) * 10
p = abs.(randn(rng, m))
q = abs.(randn(rng, n))
p .= p ./ sum(p)
q .= q ./ sum(q)

# @show norm([p;q])
# @show norm(C)


problem = OptimalTransportProblem(C, p, q) 

params = PrimalDualOptimizerParameters(
    20000,
    1e-4,
    ConstantStepsizeParams(),
)

kkt_stats_res, iter, time_basic, time_full = optimize(problem, params)

kkt_plt = Plots.plot()
Plots.plot!(
    1:length(kkt_stats_res), 
    kkt_stats_res, 
    xlabel = "Iterations", 
    ylabel = "KKT Residual", 
    yscale = :log10,
    label = false,
)

Plots.savefig(kkt_plt,joinpath("./test.png"))

