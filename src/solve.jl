"""
    solve(s::Solver, m::DRO; kwargs...)

Solves given DRO model `m` using solver `s`. 
"""
function solve(s::Solver, m::DRO; kwargs...)
    reset_stats(s, m)

    p, t, bytes, gctime, memallocs = @timed optimal(s, m.d, m; kwargs...)
    add_stats(t, bytes)

    return p
end


"""
    solve(s::Solver, m::Simplex; kwargs...)

Solves given Simplex model `m` using solver `s`. 
"""
function solve(s::Solver, m::Simplex; kwargs...)
    reset_stats(s, m)

    p, t, bytes, gctime, memallocs = @timed optimal(s, m; kwargs...)
    add_stats(t, bytes)

    return p
end
