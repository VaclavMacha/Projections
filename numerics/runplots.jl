using Revise
using Projections, Distributions, LinearAlgebra, Statistics, Plots
pyplot()

include("./Projections/numerics/plots.jl")


function find_μ_plot(μ, gμ; title::String = "title", path::String = "", type::String = "")

    plot(μ, gμ, size = (1200, 800), lw = 2)
    plot!([minimum(μ), maximum(μ)], [0, 0])

    title!(title)
    xlabel!("μ")
    ylabel!("g(μ)")

    isempty(type) || savefig(joinpath(path, string(title, ".", type)))
end


l    = 2
path = "./Projections/numerics/find_mu"
type = "svg"

## benchmark simplex_mod1
μ1 = range(1.5; stop = 2.5, length = 5)
μ2 = range(1.5; stop = 2.5, length = 1000)
g1 = find_μ_simplex_mod1(l)
find_μ_plot(μ1, g1.(μ1); title = "Simplex_mod1 (small)", path = path, type = type)
find_μ_plot(μ2, g1.(μ2); title = "Simplex_mod1 (large)", path = path, type = type)

## benchmark simplex_mod2
μ1 = range(1.5; stop = 2.5, length = 5)
μ2 = range(1.5; stop = 2.5, length = 1000)
g2 = find_μ_simplex_mod1(l)
find_μ_plot(μ1, g2.(μ1); title = "Simplex_mod2 (small)", path = path, type = type)
find_μ_plot(μ2, g2.(μ2); title = "Simplex_mod2 (large)", path = path, type = type)

## benchmark minimize_linear_on_simplex_l2
μ1 = range(0; stop = 100, length = 100)
μ2 = range(0; stop = 100, length = 1000)
g3 = find_μ_minimize_linear_on_simplex(l)
find_μ_plot(μ1, g3.(μ1); title = "Minimize_linear_on_simplex_l2 (small)", path = path, type = type)
find_μ_plot(μ2, g3.(μ2); title = "Minimize_linear_on_simplex_l2 (large)", path = path, type = type)
