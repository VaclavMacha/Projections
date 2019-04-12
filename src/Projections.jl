module Projections

using Convex, ECOS,  Roots, LinearAlgebra, Statistics


mutable struct Stats
    evals::Dict
    key::Symbol
end

stats = Stats(Dict(), :key)

function new_key!(key::Symbol)
    stats.evals[key] = 0
    stats.key = key
end
update_stats!() = stats.evals[stats.key] += 1
return_stats()  = deepcopy(stats.evals)

function reset_stats!()
    stats.evals = Dict()
    stats.key   = :key
end

include("find_root.jl")

include("simplex.jl")
include("simplex_mod1.jl")
include("simplex_mod2.jl")
include("simplex_mod3.jl")
include("simplex_mod4.jl")
include("minimize_linear_on_simplex.jl")
include("philpott.jl")

end # module
