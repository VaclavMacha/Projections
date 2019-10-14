"""
    Divergence

An abstract type covering all ϕ-divergence.
"""
abstract type Divergence end


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
    bounds(d::KullbackLeibler, m::Model)

Returns lower and upper bound for finding the root of the function `h` using the bisection method. 
"""
bounds(d::KullbackLeibler, m::Model) = (0, (m.cmax - m.cmin)/m.ε)


function h(d::KullbackLeibler, m::Model, λ::Real)
    p_hat = m.q.*exp.(m.c./λ)
    val   = p_hat' * (m.c./λ .- log(sum(p_hat)) .- m.ε)
    return isnan(val) || val == - typemax(val) ? typemax(val) : val
end


"""
    function optimal(d::KullbackLeibler, m::Model; kwargs...)  

Returns the optimal solution of the DRO model 'm' with Kullback-Leibler divergence.
"""
function optimal(d::KullbackLeibler, m::Model; kwargs...) 
    λ = bisection(d, m; kwargs...)
    p = m.q.*exp.(m.c./λ)
    return p./sum(p)
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
    bounds(d::Burg, m::Model)

Returns lower and upper bound for finding the root of the function `h` using the bisection method. 
"""
bounds(d::Burg, m::Model) = (m.cmax, m.cmax + (m.cmax - m.cmin)/m.ε)


function h(d::Burg, m::Model, λ::Real)
    val = sum(m.q.*log.(λ .- m.c)) + log(sum(m.q./(λ .- m.c))) - m.ε
    return isnan(val) ? typemax(val) : val
end


"""
    function optimal(d::Burg, m::Model; kwargs...)  

Returns the optimal solution of the DRO model 'm' with Burg entropy.
"""
function optimal(d::Burg, m::Model; kwargs...) 
    λ = bisection(d, m; kwargs...)
    p = m.q./(λ .- m.c)
    return p./sum(p)
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
    bounds(d::Hellinger, m::Model)

Returns lower and upper bound for finding the root of the function `h` using the bisection method. 
"""
bounds(d::Hellinger, m::Model) = (m.cmax, m.cmax + (2 - m.ε)*(m.cmax - m.cmin)/m.ε)


function h(d::Hellinger, m::Model, λ::Real)
    val =  2*sum(m.q./(λ .- m.c)) - (2 - m.ε)*sqrt(sum(m.q./((λ .- m.c).^2)))
    return isnan(val) ? - typemax(val) : val
end


"""
    function optimal(d::Hellinger, m::Model; kwargs...)  

Returns the optimal solution of the DRO model 'm' with Hellinger distance.
"""
function optimal(d::Hellinger, m::Model; kwargs...) 
    λ = bisection(d, m; kwargs...)
    p = m.q./(λ .- m.c).^2
    return p./sum(p)
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
    initial(d::ChiSquare, m::Model)

Returns the initial point for finding the root of the function `h` using the newton method. 
"""
function initial(d::ChiSquare, m::Model)
    f(μ)  = h(d, m, μ)
    λ_min = m.cmax
    q     = 0.01
    f_val = f(λ_min + q)

    while f_val <= 0
        f == 0 && return λ_min + q

        q    /= 10
        f_val = f(λ_min + q)
    end
    return λ_min + q
end


function h(d::ChiSquare, m::Model, λ::Real)
    val = sum(m.q .* sqrt.(λ .- m.c))*sum(m.q./sqrt.(λ .- m.c)) - 1 - m.ε
    return isnan(val) ? - typemax(val) : val
end


function ∇h(d::ChiSquare, m::Model, λ::Real)
    λc = λ .- m.c
    return (sum(m.q ./ sqrt.(λc))^2 - sum(m.q .* sqrt.(λc))*sum(m.q ./ (λc.^(3/2))))/2
end


"""
    function optimal(d::ChiSquare, m::Model; kwargs...)  

Returns the optimal solution of the DRO model 'm' with χ²-distance.
"""
function optimal(d::ChiSquare, m::Model; kwargs...) 
    λ = newton(d, m; kwargs...)
    p = m.q./sqrt.(λ .- m.c)
    return p./sum(p)
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
    initial(d::ModifiedChiSquare, m::Model)

Returns the initial point for finding the root of the function `h` using the newton method. 
"""
function initial(d::ModifiedChiSquare, m::Model)
    f(λ)  = h(d, m, λ)
    λ     = - m.cmax + 10
    f_val = f(λ)

    while f_val >= 0
        f == 0 && return λ

        λ    *= 10
        f_val = f(λ)
    end
    return λ
end


function h(d::ModifiedChiSquare, m::Model, λ::Real)
    val = sum(m.q .* max.(λ .+ m.c, 0).^2) - (1 + m.ε) * (sum(m.q .* max.(λ .+ m.c, 0)))^2
    return isnan(val) ? - typemax(val) : val
end


function ∇h(d::ModifiedChiSquare, m::Model, λ::Real)
    I = findall(λ .+ m.c .>= 0) 
    return 2 * sum(m.q[I] .* (λ .+ m.c[I])) * (1 - (1 + m.ε) * sum(m.q[I]))
end


"""
    function optimal(d::ModifiedChiSquare, m::Model; kwargs...)  

Returns the optimal solution of the DRO model 'm' with modified χ²-distance.
"""
function optimal(d::ModifiedChiSquare, m::Model; kwargs...) 
    λ = newton(d, m; kwargs...)
    p = m.q.*max.(λ .+ m.c, 0)
    return p./sum(p)
end