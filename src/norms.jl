"""
  Norm

An abstract type covering all norms.
"""
abstract type Norm end


"""
  solve_exact(d::Norm, m::Model)

Solves the given model `m` with norm `d` using the ECOS solver. 
"""
function solve_exact(d::Norm, m::Model)
    k = normtype(d)
    p = Convex.Variable(length(m.q))

    objective   = m.c'*p
    constraints = [sum(p) == 1,
                   p >= 0,
                   Convex.norm(p - m.q, k) <= m.ε]

    problem = Convex.maximize(objective, constraints)
    Convex.solve!(problem, ECOS.ECOSSolver(verbose = false), verbose = false)
    return vec(p.value)
end


# ----------------------------------------------------------------------------------------------------------
# l-∞ norm
# ----------------------------------------------------------------------------------------------------------
"""
  Linf

An empty structure representing l-infinity norm.
"""
struct Linf <: Norm end


"""
  name(d::Linf)

Returns the full name of the l-infinity norm.
"""
name(d::Linf) = "l-infinity norm"


"""
  normtype(d::Linf)

Returns the type of l-infinity norm.
"""
normtype(d::Linf) = Inf


"""
  solve(d::Linf, m::Model)

Solves the given model `m` with l-infinity norm using our new approach. 
"""
function solve(d::Linf, m::Model)
    c      = .- m.c
    perm   = sortperm(c)
    c      = c[perm]
    p      = m.q[perm]
    i      = 1
    j      = length(c)
    δ_down = 0

    while i < j
        δ1    = min(1 - p[i], m.ε)
        δ2    = 0
        p[i] += δ1

        while δ2 < δ1
            if min(p[j], m.ε - δ_down) >= δ1 - δ2
                p[j]   -= δ1 - δ2
                δ_down += δ1 - δ2
                break
            else
                δ2     += min(p[j], m.ε - δ_down)
                p[j]   -= min(p[j], m.ε - δ_down)
                δ_down  = 0
                j      -= 1

                if i == j
                    p[i] += - δ1 + δ2
                    break
                end
            end
        end
        i += 1
    end
    return p[invperm(perm)]
end


# ----------------------------------------------------------------------------------------------------------
# l-1 norm
# ----------------------------------------------------------------------------------------------------------
"""
  Lone

An empty structure representing l-1 norm.
"""
struct Lone <: Norm end


"""
  name(d::Lone)

Returns the full name of the l-1 norm.
"""
name(d::Lone) = "l-1 norm"


"""
  normtype(d::Lone)

Returns the type of l-1 norm.
"""
normtype(d::Lone) = 1


"""
  solve(d::Lone, m::Model)

Solves the given model `m` with l-1 norm using our new approach. 
"""
function solve(d::Lone, m::Model)
    c    = .- m.c
    perm = sortperm(c)
    c    = c[perm]
    p    = m.q[perm]
    i    = 1
    j    = length(c)
    δ_up = 0

    while i < j && δ_up < m.ε/2
        δ1   = min(1 - p[i], m.ε/2-δ_up)
        δ_up += δ1
        δ2   = 0
        p[i] += δ1

        while δ2 < δ1
            if p[j] >= δ1 - δ2
                p[j] += - δ1 + δ2
                break
            else
                δ2 = δ2 + p[j]
                p[j] = 0
                j -= 1
                if i == j
                    p[i] += - δ1 + δ2
                    break
                end
            end
        end
        i += 1
    end
    return p[invperm(perm)]
end


# ----------------------------------------------------------------------------------------------------------
# l-2 norm
# ----------------------------------------------------------------------------------------------------------
"""
  Ltwo

An empty structure representing l-2 norm.
"""
struct Ltwo <: Norm end


"""
  name(d::Ltwo)

Returns the full name of the l-2 norm.
"""
name(d::Ltwo) = "l-2 norm"


"""
  normtype(d::Ltwo)

Returns the type of l-2 norm.
"""
normtype(d::Ltwo) = 2


bounds(d::Ltwo, m::Model) = (1e-10, Inf)

function g(d::Ltwo, m::Model, μ::Real)
    n     = length(m.q)
    s     = sort(μ.*m.q .+ m.c)
    λ     = 0
    g     = μ
    g_old = g

    for i in n-1:-1:1
        g_old = g
        g    -= (n - i)*(s[i + 1] - s[i])
        if g <= 0
            λ = s[i] - (s[i + 1] - s[i])/(g_old - g)*g
            break
        end
    end
    if g > 0
        λ = Statistics.mean(m.c)
    end
    return λ
end

function h(d::Ltwo, m::Model, μ::Real)
    λ = g(d, m, μ)
    return sum((min.(λ .- m.c, μ .* m.q)).^2) - m.ε^2 * μ^2
end


function optimal(d::Ltwo, m::Model, μ::Real) 
    λ = g(d, m, μ)
    return max.(m.q .- (λ .- m.c)./μ, 0)
end


"""
  solve(d::Ltwo, m::Model)

Solves the given model `m` with l-2 norm using our new approach. 
"""
function solve(d::Ltwo, m::Model)
    Ilen = length(m.Imax)
    Isum = sum(m.q[m.Imax])
    p    = zero(m.q)
    p[m.Imax] .= m.q[m.Imax] .+ 1/Ilen .- Isum/Ilen

    if sum(abs2.(p .- m.q)) <= m.ε^2
      return p
    else
      μ = Roots.bisection(μ -> h(d, m, μ), bounds(d,m)...)
      return optimal(d, m, μ)
    end
end