module Projections

using Convex, SCS, LinearAlgebra, Roots, Statistics

include("simplex.jl")
include("simplex_mod1.jl")
include("simplex_mod2.jl")
include("minimize_linear_on_simplex.jl")

include("projection_exact.jl")
include("projection.jl")

end # module
