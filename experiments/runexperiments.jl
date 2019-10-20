using Projections, CSV, DataFrames, Plots, Distributions, LinearAlgebra, Random

path     = @__DIR__
savepath = path

include(joinpath(path, "utilities.jl"))

@time table1 = comparison_divergences(;save = true, savepath = savepath);
@time table2 = comparison_norms(;save = true, savepath = savepath);
@time table4 = comparison_philpott(;save = true, savepath = savepath);
@time table3 = comparison_general_solvers(;save = true, savepath = savepath);


## Kullback-Leibler divergence
m = get_DRO(10, KullbackLeibler(), 0.1)
λ = 1:0.01:3
h(λ,m; save = true, savepath = savepath)

## Burg entropy
m = get_DRO(10, Burg(), 0.1)
λ = 1.9:0.01:3
h(λ,m; save = true, savepath = savepath)

## Hellinger distance
m = get_DRO(10, Hellinger(), 0.1)
λ = 1.9:0.01:5
h(λ,m; save = true, savepath = savepath)

## χ²-distance distance
m = get_DRO(10, ChiSquare(), 0.1)
λ = 1.9:0.01:3
h(λ,m; save = true, savepath = savepath)

## Modified χ²-distance distance
m = get_DRO(10, ModifiedChiSquare(), 0.1)
λ = -1.8:0.01:3
h(λ,m; save = true, savepath = savepath)

## l-2 norm
m = get_DRO(10, Ltwo(), 0.1)
λ = 0:0.01:30
h(λ,m; save = true, savepath = savepath)