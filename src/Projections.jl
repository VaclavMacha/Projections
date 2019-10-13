module Projections

import JuMP, Ipopt, Convex, ECOS, Roots, LinearAlgebra, Statistics

export Model, solve, solve_exact 
export Divergence, KullbackLeibler, Burg, Hellinger, ChiSquare, ModifiedChiSquare
export Norm, Linf, Lone, Ltwo, Philpott 

include("model.jl")
include("divergences.jl")
include("norms.jl")
include("phillpott.jl")
include("findroot.jl")

end # module