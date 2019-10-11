module Projections

import JuMP, Ipopt, Convex, ECOS, Roots, LinearAlgebra, Statistics

include("model.jl")
include("divergences.jl")
include("norms.jl")
include("phillpott.jl")

end # module
