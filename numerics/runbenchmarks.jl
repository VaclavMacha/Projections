using Revise
using Projections, Distributions, LinearAlgebra, Statistics, Plots
pyplot()

include("./Projections/numerics/benchmarks.jl")


function benchmark_plot(n,t, err, cnt = Float64[]; title::String   = "title",
                                                   scale::Function = identity,
                                                   path::String    = "",
                                                   type::String    = "")
    isempty(cnt) ? t_tmp = t     : t_tmp = t./cnt
    isempty(cnt) ? err_tmp = err : err_tmp = err./std(cnt)

    plot(n, scale.(t_tmp), yerror = scale.(err_tmp), size = (1200, 800))
    title!(title)
    xlabel!("n")
    ylabel!("t [s]")

    isempty(type) || savefig(joinpath(path, string(title, ".", type)))
end

l = 1:0.25:7
n = ceil.(Int64, 10 .^ l)
path = "./Projections/numerics/times"
type = "svg"


## benchmark simplex
@time t1, std1 = benchmark_simplex(l);
benchmark_plot(n, t1, std1; title = "Simplex", path = path, type = type)

## benchmark simplex_mod1
@time t2, std2, cnt2 = benchmark_simplex_mod1(l);
benchmark_plot(n, t2, std2, cnt2; title = "Simplex_mod1", path = path, type = type, max_rep = 20)

## benchmark simplex_mod2
@time t3, std3, cnt3 = benchmark_simplex_mod2(l);
benchmark_plot(n, t3, std3, cnt3; title = "Simplex_mod2", path = path, type = type, max_rep = 20)

## benchmark minimize_linear_on_simplex_lInf
@time t4, std4 = benchmark_minimize_linear_on_simplex(l, Inf);
benchmark_plot(n, t4, std4; title = "Minimize_linear_on_simplex_lInf", path = path, type = type)

## benchmark minimize_linear_on_simplex_l1
@time t5, std5 = benchmark_minimize_linear_on_simplex(l, 1);
benchmark_plot(n, t5, std5; title = "Minimize_linear_on_simplex_l1", path = path, type = type)

## benchmark minimize_linear_on_simplex_l2
@time t6, std6, cnt6 = benchmark_minimize_linear_on_simplex(l, 2);
benchmark_plot(n, t6, std6, cnt6; title = "minimize_linear_on_simplex_l2", path = path, type = type)
