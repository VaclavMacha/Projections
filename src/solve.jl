# ----------------------------------------------------------------------------------------------------------
# ϕ-divergences
# ----------------------------------------------------------------------------------------------------------
"""
    solve(d::Divergence, m::Model; kwargs...)

Solves the given model `m` with ϕ-divergence `d` using our new approach. 
"""
function solve(d::Divergence, m::Model; kwargs...)
    p = zero(m.q)
    p[m.Imax] .= m.q[m.Imax]
    p ./= sum(m.q[m.Imax])

    ϕ = generate(d)

    if sum(m.q .* ϕ.(p ./ m.q)) <= m.ε
      return p
    else
      return optimal(d, m; kwargs...)
    end
end


# ----------------------------------------------------------------------------------------------------------
# p-norms
# ----------------------------------------------------------------------------------------------------------
"""
    solve(d::Union{Linf, Lone}, m::Model; kwargs...) 

Solves the given model `m` with l-infinity or l-1 norm  using our new approach.     
"""
function solve(d::Union{Linf, Lone}, m::Model; kwargs...)
    return optimal(d, m)
end


"""
    solve(d::Union{Linf, Lone, Philpott}, m::Model; kwargs...) 

Solves the given model `m` with l-2 norm using Philpott's algorithm.     
"""
function solve(d::Philpott, m::Model; kwargs...)
    return optimal(d, m; kwargs...)
end



"""
    solve(d::Ltwo, m::Model; kwargs...)

Solves the given model `m` with l-2 norm using our new approach. 
"""
function solve(d::Ltwo, m::Model; kwargs...)
    Ilen = length(m.Imax)
    Isum = sum(m.q[m.Imax])
    p    = zero(m.q)
    p[m.Imax] .= m.q[m.Imax] .+ 1/Ilen .- Isum/Ilen

    if sum(abs2, p - m.q) <= m.ε^2
      return p
    else
      return optimal(d, m; kwargs...)
    end
end