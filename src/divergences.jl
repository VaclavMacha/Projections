function optimal_simple(d::Divergence, m::DRO)
    p = zero(m.q)
    p[m.Imax] .= m.q[m.Imax]
    return p./sum(m.q[m.Imax])
end


# ----------------------------------------------------------------------------------------------------------
# Kullback-Leibler divergence
# ----------------------------------------------------------------------------------------------------------
"""
    KullbackLeibler

An empty structure representing Kullback-Leibler divergence.
"""
struct KullbackLeibler <: Divergence end


"""
    generate(d::KullbackLeibler)

Returns the generating function `ϕ(t) = t⋅log(t)` of the Kullback-Leibler divergence.
The generating function for `t == 0` is defined as `0`.
"""
generate(d::KullbackLeibler) = ϕ(t) = iszero(t) ? zero(t) : t*log(t)


"""
    name(d::KullbackLeibler)

Returns the full name of the Kullback-Leibler divergence.
"""
name(d::KullbackLeibler) = "Kullback-Leibler divergence"


"""
    bounds(d::KullbackLeibler, m::DRO)

Returns lower and upper bound for finding the root of the function `h` using the bisection method. 
"""
bounds(d::KullbackLeibler, m::DRO) = (0, (m.cmax - m.cmin)/m.ε)


function h(d::KullbackLeibler, m::DRO, λ::Real)
    p_hat = m.q.*exp.(m.c./λ)
    val   = p_hat' * (m.c./λ .- log(sum(p_hat)) .- m.ε)
    return isnan(val) || val == - typemax(val) ? typemax(val) : val
end


"""
    function optimal(s::Sadda, d::KullbackLeibler, m::DRO; kwargs...)  

Returns the optimal solution of the DRO model 'm' with Kullback-Leibler divergence.
"""
function optimal(s::Sadda, d::KullbackLeibler, m::DRO; kwargs...) 
    p = optimal_simple(d, m)
    ϕ = generate(d)

    if sum(m.q .* ϕ.(p ./ m.q)) <= m.ε
        return p
    else
        λ = bisection(d, m; kwargs...)
        p = m.q.*exp.(m.c./λ)
        return p./sum(p)
    end
end


# ----------------------------------------------------------------------------------------------------------
# Burg entropy
# ----------------------------------------------------------------------------------------------------------
"""
    Burg

An empty structure representing Burg entropy.
"""
struct Burg <: Divergence end


"""
    generate(d::Burg, t)

Returns the generating function `ϕ(t) = - log(t)` of the Burg entropy.
"""
generate(d::Burg) = ϕ(t) = - log(max(t,0))


"""
    name(d::Burg)

Returns the full name of the Burg entropy.
"""
name(d::Burg) = "Burg entropy"


"""
    bounds(d::Burg, m::DRO)

Returns lower and upper bound for finding the root of the function `h` using the bisection method. 
"""
bounds(d::Burg, m::DRO) = (m.cmax, m.cmax + (m.cmax - m.cmin)/m.ε)


function h(d::Burg, m::DRO, λ::Real)
    val = sum(m.q.*log.(λ .- m.c)) + log(sum(m.q./(λ .- m.c))) - m.ε
    return isnan(val) ? typemax(val) : val
end


"""
    function optimal(s::Sadda, d::Burg, m::DRO; kwargs...)  

Returns the optimal solution of the DRO model 'm' with Burg entropy.
"""
function optimal(s::Sadda, d::Burg, m::DRO; kwargs...) 
    p = optimal_simple(d, m)
    ϕ = generate(d)

    if sum(m.q .* ϕ.(p ./ m.q)) <= m.ε
        return p
    else
        λ = bisection(d, m; kwargs...)
        p = m.q./(λ .- m.c)
        return p./sum(p)
    end
end


# ----------------------------------------------------------------------------------------------------------
# Hellinger distance
# ----------------------------------------------------------------------------------------------------------
"""
    Hellinger

An empty structure representing Hellinger distance.
"""
struct Hellinger <: Divergence end


"""
    generate(d::Hellinger, t)

Returns the generating function `ϕ(t) = (√t - 1)²`  of the Hellinger distance.
"""
generate(d::Hellinger) = ϕ(t) = (sqrt(max(t,0)) - 1)^2


"""
    name(d::Hellinger)

Returns the full name of the Hellinger distance.
"""
name(d::Hellinger) = "Hellinger distance"


"""
    bounds(d::Hellinger, m::DRO)

Returns lower and upper bound for finding the root of the function `h` using the bisection method. 
"""
bounds(d::Hellinger, m::DRO) = (m.cmax, m.cmax + (2 - m.ε)*(m.cmax - m.cmin)/m.ε)


function h(d::Hellinger, m::DRO, λ::Real)
    val =  2*sum(m.q./(λ .- m.c)) - (2 - m.ε)*sqrt(sum(m.q./((λ .- m.c).^2)))
    return isnan(val) ? - typemax(val) : val
end


"""
    function optimal(s::Sadda, d::Hellinger, m::DRO; kwargs...)  

Returns the optimal solution of the DRO model 'm' with Hellinger distance.
"""
function optimal(s::Sadda, d::Hellinger, m::DRO; kwargs...) 
    p = optimal_simple(d, m)
    ϕ = generate(d)

    if sum(m.q .* ϕ.(p ./ m.q)) <= m.ε
        return p
    else
        λ = bisection(d, m; kwargs...)
        p = m.q./(λ .- m.c).^2
        return p./sum(p)
    end
end


# ----------------------------------------------------------------------------------------------------------
# χ²-distance
# ----------------------------------------------------------------------------------------------------------
"""
    ChiSquare

An empty structure representing χ²-distance.
"""
struct ChiSquare <: Divergence end


"""
    generate(d::ChiSquare, t)

Returns the generating function `ϕ(t) = (t - 1)²/t`  of the χ²-distance.
"""
generate(d::ChiSquare) = ϕ(t) = ((t - 1)^2)/t


"""
    name(d::ChiSquare)

Returns the full name of the χ²-distance.
"""
name(d::ChiSquare) = "χ²-distance"


"""
    initial(d::ChiSquare, m::DRO)

Returns the initial point for finding the root of the function `h` using the newton method. 
"""
function initial(d::ChiSquare, m::DRO)
    f(μ)  = h(d, m, μ)
    λ_min = m.cmax
    q     = 0.01
    f_val = f(λ_min + q)
    add_eval()

    while f_val <= 0
        f == 0 && return λ_min + q

        q    /= 10
        f_val = f(λ_min + q)
        add_eval()
    end
    return λ_min + q
end


function h(d::ChiSquare, m::DRO, λ::Real)
    val = sum(m.q .* sqrt.(λ .- m.c))*sum(m.q./sqrt.(λ .- m.c)) - 1 - m.ε
    return isnan(val) ? - typemax(val) : val
end


function ∇h(d::ChiSquare, m::DRO, λ::Real)
    λc = λ .- m.c
    return (sum(m.q ./ sqrt.(λc))^2 - sum(m.q .* sqrt.(λc))*sum(m.q ./ (λc.^(3/2))))/2
end


"""
    function optimal(s::Sadda, d::ChiSquare, m::DRO; kwargs...)  

Returns the optimal solution of the DRO model 'm' with χ²-distance.
"""
function optimal(s::Sadda, d::ChiSquare, m::DRO; kwargs...) 
    p = optimal_simple(d, m)
    ϕ = generate(d)

    if sum(m.q .* ϕ.(p ./ m.q)) <= m.ε
        return p
    else
        λ = newton(d, m; kwargs...)
        p = m.q./sqrt.(λ .- m.c)
        return p./sum(p)
    end
end


# ----------------------------------------------------------------------------------------------------------
# Modified χ²-distance
# ----------------------------------------------------------------------------------------------------------
"""
    ModifiedChiSquare

An empty structure representing modified χ²-distance.
"""
struct ModifiedChiSquare <: Divergence end


"""
    generate(d::ModifiedChiSquare, t)

Returns the generating function `ϕ(t) = (t - 1)²`  of the modified χ²-distance.
"""
generate(d::ModifiedChiSquare) = ϕ(t) = (t - 1)^2


"""
    name(d::ModifiedChiSquare)

Returns the full name of the modified χ²-distance.
"""
name(d::ModifiedChiSquare) = "Modified χ²-distance"


"""
    initial(d::ModifiedChiSquare, m::DRO)

Returns the initial point for finding the root of the function `h` using the newton method. 
"""
function initial(d::ModifiedChiSquare, m::DRO)
    f(λ)  = h(d, m, λ)
    λ     = - m.cmax + 10
    f_val = f(λ)
    add_eval()

    while f_val >= 0
        f == 0 && return λ

        λ    *= 10
        f_val = f(λ)
        add_eval()
    end
    return λ
end


function h(d::ModifiedChiSquare, m::DRO, λ::Real)
    val = sum(m.q .* max.(λ .+ m.c, 0).^2) - (1 + m.ε) * (sum(m.q .* max.(λ .+ m.c, 0)))^2
    return isnan(val) ? - typemax(val) : val
end


function ∇h(d::ModifiedChiSquare, m::DRO, λ::Real)
    I = findall(λ .+ m.c .>= 0) 
    return 2 * sum(m.q[I] .* (λ .+ m.c[I])) * (1 - (1 + m.ε) * sum(m.q[I]))
end


"""
    function optimal(s::Sadda, d::ModifiedChiSquare, m::DRO; kwargs...)  

Returns the optimal solution of the DRO model 'm' with modified χ²-distance.
"""
function optimal(s::Sadda, d::ModifiedChiSquare, m::DRO; kwargs...)
    p = optimal_simple(d, m)
    ϕ = generate(d)

    if sum(m.q .* ϕ.(p ./ m.q)) <= m.ε
        return p
    else
        λ = newton(d, m; kwargs...)
        p = m.q.*max.(λ .+ m.c, 0)
        return p./sum(p)
    end
end