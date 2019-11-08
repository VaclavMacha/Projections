"""
    solve(s::Solver, m::DRO; kwargs...)

Solves given DRO model `m` using solver `s`. 
"""
function solve(s::S, m::DRO; kwargs...) where {S<:Solver}
    reset_stats(s, m)

    @assert S <: Philpott && typeof(m.d) <: Ltwo "Philpott solver is supported only for the DRO model with l2 norm"

    p, t, bytes, gctime, memallocs = @timed optimal(s, m.d, m; kwargs...)
    add_stats(t, bytes)

    return p
end


"""
    solve(s::Solver, m::Simplex; kwargs...)

Solves given Simplex model `m` using solver `s`. 
"""
function solve(s::S, m::Simplex; kwargs...) where {S<:Solver}
    reset_stats(s, m)

    @assert !(S <: Philpott) "Philpott solver is supported only for the DRO model with l2 norm"

    p, t, bytes, gctime, memallocs = @timed optimal(s, m; kwargs...)
    add_stats(t, bytes)

    return p
end
