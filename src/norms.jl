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
    optimal(s::Our, d::Linf, m::DRO; kwargs...)

Returns the optimal solution of the DRO model 'm' with l-infinity norm. 
"""
function optimal(s::Our, d::Linf, m::DRO)
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
    optimal(s::Our, d::Lone, m::DRO)

Returns the optimal solution of the DRO model 'm' with l-1 norm.
"""
function optimal(s::Our, d::Lone, m::DRO; kwargs...)
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


"""
    initial(d::Ltwo, m::DRO)

Returns the initial point for finding the root of the function `h` using the newton method. 
"""
function initial(d::Ltwo, m::DRO)
    f(μ)  = h(d, m, μ)
    μ     = 10
    f_val = f(μ)
    add_eval()

    while f_val >= 0
        f == 0 && return μ

        μ    *= 10
        f_val = f(μ)
        add_eval()
    end
    return μ
end


function g(d::Ltwo, m::DRO, μ::Real)
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


function h(d::Ltwo, m::DRO, μ::Real; λ::Real = g(d, m, μ))
    return sum((min.(λ .- m.c, μ .* m.q)).^2) - m.ε^2 * μ^2
end


function ∇h(d::Ltwo, m::DRO, μ::Real; λ::Real = g(d, m, μ))
    q_i = @view m.q[λ .- m.c .>  μ .* m.q]
    c_i = @view m.c[λ .- m.c .<= μ .* m.q]

    return 2*μ*(sum(abs2, q_i) - m.ε^2) - 2*sum(q_i)*sum(λ .- c_i)/length(c_i)
end


"""
    optimal(s::Our, d::Ltwo, m::DRO; kwargs...) 

Returns the optimal solution of the DRO model 'm' with l-2 norm.
"""
function optimal(s::Our, d::Ltwo, m::DRO; kwargs...)
    Ilen = length(m.Imax)
    Isum = sum(m.q[m.Imax])
    p    = zero(m.q)
    p[m.Imax] .= m.q[m.Imax] .+ 1/Ilen .- Isum/Ilen

    if sum(abs2, p - m.q) <= m.ε^2
        return p
    else
        μ = newton(d, m; kwargs...) 
        λ = g(d, m, μ)
        return max.(m.q .- (λ .- m.c)./μ, 0)
    end
end


"""
    optimal(s::Philpott, d::Philpott, m::DRO; atol::Real = 1e-10) 

Returns the optimal solution given by Philpott's algorithm of the DRO model 'm' with l-2 norm.
"""
function optimal(s::Philpott, d::Ltwo, m::DRO; atol::Real = 1e-10, kwargs...)
    n      = length(m.q)
    c_mean = Statistics.mean(m.c)
    c_std  = Statistics.stdm(m.c, c_mean; corrected = false)
    p      = m.q .+ m.ε .* (m.c .- c_mean) ./ (sqrt(n) * c_std)

    isapprox(c_std, 0, atol = atol)         && return m.q
    m.ε <= sqrt(n / (n - 1)) * minimum(m.q) && return p

    I  = collect(1:n)
    j  = 0
    a1 = 0
    a2 = 0
    a3 = sqrt(n)*m.ε

    @views for I_len in n-1:-1:1

        minimum(p[I]) >= 0 && return p

        ## find index j
        J   = I[p[I] .< 0]
        ε_p = @. (a1^2 + (c_std * (I_len * m.q[J] + a1) / (m.c[J] - c_mean))^2) / I_len + a2
        j   = J[findmin(ε_p)[2]]

        deleteat!(I, searchsortedfirst(I, j))
        p[j] = 0

        ## update p
        a1 += m.q[j]
        a2 += m.q[j]^2
        a3 = sqrt(I_len*(m.ε^2 - a2) - a1^2)

        c_mean = ((I_len + 1) * c_mean - m.c[j]) / I_len
        c_std  = Statistics.stdm(m.c[I], c_mean; corrected = false)

        @. p[I] = m.q[I] + (a1 + a3 * (m.c[I] - c_mean) / c_std) / I_len

        if isapprox(c_std, 0, atol = atol)
            @. p[I] = m.q[I] + a1 / I_len
            return p
        end
    end
    p[I] = 1
    return p
end