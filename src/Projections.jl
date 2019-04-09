module Projections

using Convex, ECOS,  Roots, LinearAlgebra, Statistics

tmp = Dict(:count => 0)
update_count!() = tmp[:count] += 1
reset_count!()  = tmp[:count] = 0
get_count()     = tmp[:count]

include("simplex.jl")
include("simplex_mod1.jl")
include("simplex_mod2.jl")
include("minimize_linear_on_simplex.jl")

include("projection_exact.jl")
include("projection.jl")

end # module
