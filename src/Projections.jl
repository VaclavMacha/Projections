module Projections

using Convex, ECOS,  Roots, LinearAlgebra, Statistics


mutable struct Stats
    secant::Integer
    bisection::Integer
    evals::Integer
    key::Symbol
end

stats = Stats(0, 0, 0, :secant)

update_stats!()          = setfield!(stats, stats.key, getfield(stats, stats.key) + 1)
update_key!(key::Symbol) = setfield!(stats, :key, key)

function get_stats()
    setfield!(stats, :evals, stats.secant + stats.bisection)
    return stats
end

function reset_stats!()
    stats.secant = 0
    stats.bisection = 0
    stats.evals = 0
    stats.key = :secant
end


include("simplex.jl")
include("simplex_mod1.jl")
include("simplex_mod2.jl")
include("minimize_linear_on_simplex.jl")

include("projection_exact.jl")
include("projection.jl")

end # module
