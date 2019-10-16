mutable struct Stats
    model::String
    constraint::String
    optimizer::String
    evals::Int
    evaltime::Real
    bytes::Real
end
 

local stats
stats = Stats("model", "constraint", "optimizer", 0, 0, 0)


function reset_stats(d::Union{KullbackLeibler, Burg, Hellinger})
    stats.model      = "DRO"
    stats.constraint = string(typeof(d).name)
    stats.optimizer  = "bisection"
    stats.evals      = 0
    stats.evaltime   = 0
    stats.bytes      = 0
end


function reset_stats(d::Union{ChiSquare, ModifiedChiSquare, Ltwo})
    stats.model      = "DRO"
    stats.constraint = string(typeof(d).name)
    stats.optimizer  = "newton"
    stats.evals      = 0
    stats.evaltime   = 0
    stats.bytes      = 0
end


function reset_stats(d::Union{Linf, Lone, Philpott})
    stats.model      = "DRO"
    stats.constraint = string(typeof(d).name)
    stats.optimizer  = "none"
    stats.evals      = 0
    stats.evaltime   = 0
    stats.bytes      = 0
end


function add_eval(k::Int = 1)
    stats.evals += k
end


function add_stats(t::Real, b::Real)
    stats.evaltime = t
    stats.bytes    = b
end