module Projections

import LinearAlgebra, Statistics, DataFrames, ProgressMeter
import JuMP, Ipopt, CPLEX
import Convex, ECOS

export DRO, Simplex, Simplex1, Simplex2, solve
export Solver, Sadda, General, Philpott
export Constraint 
export Divergence, KullbackLeibler, Burg, Hellinger, ChiSquare, ModifiedChiSquare
export Norm, Linf, Lone, Ltwo, Philpott 

abstract type Model end
abstract type Simplex <: Model end

abstract type Solver end
struct Sadda <: Solver end
struct General <: Solver end
struct Philpott <: Solver end

abstract type Constraint end
abstract type Divergence <: Constraint end
abstract type Norm <: Constraint end

include("model.jl")
include("divergences.jl")
include("norms.jl")
include("simplex.jl")
include("utilities.jl")
include("findroot.jl")
include("generalsolvers.jl")
include("solve.jl")
include("benchmarks.jl")

end # module