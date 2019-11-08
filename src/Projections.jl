module Projections

import LinearAlgebra, Statistics, DataFrames, ProgressMeter
import JuMP, Ipopt, CPLEX
import Convex, ECOS

export DRO, Simplex, solve, benchmark
export Solver, Our, General, Philpott
export Constraint 
export Divergence, KullbackLeibler, Burg, Hellinger, ChiSquare, ModifiedChiSquare
export Norm, Linf, Lone, Ltwo, Philpott 

abstract type Model end

abstract type Solver end
struct Our <: Solver end
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