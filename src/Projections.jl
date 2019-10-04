module Projections

using Convex, ECOS,  Roots, LinearAlgebra, Statistics, Distributions


mutable struct Stats
    evals::Dict
    stats::Dict
    key::Symbol
end

stats = Stats(Dict{Symbol, Int64}(), Dict{Symbol, Any}(), :key)

change_key!(key::Symbol) = stats.key = key

function add_eval!()
    if haskey(stats.evals, stats.key)
        stats.evals[stats.key] += 1
    else
        stats.evals[stats.key]  = 1
    end
    return nothing
end

add_stat!(args::Pair...) = [stats.stats[key] = val for (key, val) in args]

return_evals() = deepcopy(stats.evals)
return_stats() = deepcopy(stats.stats)

function reset_stats!()
    stats.evals = Dict{Symbol, Int64}()
    stats.stats = Dict{Symbol, Any}()
    stats.key   = :key
end

include("find_root.jl")

include("simplex.jl")
include("DRO.jl")
include("philpott.jl")

end # module
