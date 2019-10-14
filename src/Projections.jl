module Projections

import JuMP, Ipopt, Convex, ECOS, LinearAlgebra, Statistics

export Model, solve, generalsolve 
export Divergence, KullbackLeibler, Burg, Hellinger, ChiSquare, ModifiedChiSquare
export Norm, Linf, Lone, Ltwo, Philpott 

include("model.jl")
include("divergences.jl")
include("norms.jl")
include("findroot.jl")
include("solve.jl")
include("generalsolve.jl")

end # module