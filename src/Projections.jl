module Projections

import JuMP, Ipopt, Roots, LinearAlgebra

include("model.jl")
include("divergence.jl")
include("solve.jl")

end # module
