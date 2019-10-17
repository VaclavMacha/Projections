mutable struct Stats
    model::String
    constraint::String
    solver::String
    optimizer::String
    evals::Int
    evaltime::Real
    bytes::Real
end
 

local stats
stats = Stats("model", "constraint", "solver", "optimizer", 0, 0, 0)


function reset_stats(s::Solver, m::DRO)
    stats.model      = "DRO"
    stats.constraint = string(typeof(m.d).name)
    stats.solver     = string(typeof(s).name)
    stats.optimizer  = "none"
    stats.evals      = 0
    stats.evaltime   = 0
    stats.bytes      = 0
    return 
end


function add_eval(k::Int = 1)
    stats.evals += k
end


function add_stats(t::Real, b::Real)
    stats.evaltime = t
    stats.bytes    = b
end