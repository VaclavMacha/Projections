mutable struct Stats
    model::String
    method::String
    evals::Int
    evaltime::Real
    bytes::Real
end
 

local stats
stats = Stats("model", "method", 0, 0, 0)


function reset_stats(d::Union{KullbackLeibler, Burg, Hellinger})
    stats.model    = string(typeof(d).name)
    stats.method   = "bisection"
    stats.evals    = 0
    stats.evaltime = 0
    stats.bytes    = 0
end


function reset_stats(d::Union{ChiSquare, ModifiedChiSquare, Ltwo})
    stats.model  = string(typeof(d).name)
    stats.method = "newton"
    stats.evals  = 0
end


function reset_stats(d::Union{Linf, Lone, Philpott})
    stats.model  = string(typeof(d).name)
    stats.method = "none"
    stats.evals  = 0
end


function add_eval(k::Int = 1)
    stats.evals += k
end


function add_stats(t::Real, b::Real)
    stats.evaltime = t
    stats.bytes    = b
end