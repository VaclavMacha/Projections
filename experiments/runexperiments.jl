using Projections, CSV, DataFrames, Plots, Distributions, LinearAlgebra, Random

path     = @__DIR__
savepath = path

include(joinpath(path, "experiments.jl"))

## initialization
save = false

l1 = unique(vcat(2:0.1:3, 3:0.05:6));
l2 = unique(vcat(2:0.1:3, 3:0.05:4));
N1 = @. ceil(Int64, 10^l1)
N2 = @. ceil(Int64, 10^l2)

## benchmarks
comparison_solver(Our(), N1; save = false, maxevals = 1);
@time table1 = comparison_solver(Our(), N1; save = save,  savepath = savepath, maxevals = 100);

comparison_solver(General(), N2; save = false, maxevals = 1);
@time table2 = comparison_solver(General(), N2; save = save,  savepath = savepath, maxevals = 100);

comparison_philpott(N2; save = false, maxevals = 1);
@time table3 = comparison_philpott(N2; save = save, savepath = savepath, maxevals = 100);

## tables
savetable(vcat(table1, table2, table3), save = save, savepath = savepath, savename = "comparison") 

## Kullback-Leibler divergence
m = model_DRO(10, KullbackLeibler(), 0.1)
λ = 1:0.01:3
h(λ,m; save = save, savepath = savepath)

## Burg entropy
m = model_DRO(10, Burg(), 0.1)
λ = 1.9:0.01:3
h(λ, m; save = save, savepath = savepath)

## Hellinger distance
m = model_DRO(10, Hellinger(), 0.1)
λ = 1.9:0.01:5
h(λ, m; save = save, savepath = savepath)

## χ²-distance distance
m = model_DRO(10, ChiSquare(), 0.1)
λ = 1.9:0.01:3
h(λ, m; save = save, savepath = savepath)

## Modified χ²-distance distance
m = model_DRO(10, ModifiedChiSquare(), 0.1)
λ = -1.8:0.01:3
h(λ, m; save = save, savepath = savepath)

## l-2 norm
m = model_DRO(10, Ltwo(), 0.1)
λ = 0:0.01:30
h(λ, m; save = save, savepath = savepath)