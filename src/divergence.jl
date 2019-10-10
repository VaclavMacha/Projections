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
  check_ε(d::KullbackLeibler, m::Model)

Returns `true` if the constraint for the ε parameter for the Kullback-Leibler divergence is met
and `false` otherwise.
"""
check_ε(d::KullbackLeibler, m::Model) = m.ε < -log(sum(m.q[m.Imax]))


function find_mu(d::KullbackLeibler, m::Model)
  μmin = 0
  μmax = (m.cmax - m.cmin)/m.ε

  function h(μ)
    p_hat = m.q.*exp.(m.c./μ)
    val   = p_hat' * (m.c./μ .- log(sum(p_hat)) .- m.ε)
    return isnan(val) || val == -typemax(val) ? typemax(val) : val
  end

  return h, (μmin, μmax)
end


function optimal(d::KullbackLeibler, m::Model, μ::Real) 
  p = m.q.*exp.(m.c./μ)
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
  check_ε(d::Burg, m::Model)

Returns `true` if the constraint for the ε parameter for the Burg entropy is met
and `false` otherwise.
"""
check_ε(d::Burg, m::Model) = true


function find_mu(d::Burg, m::Model)
  μmin = m.cmax
  μmax = m.cmax + (m.cmax - m.cmin)/m.ε

  function h(μ)
    val = sum(m.q.*log.(μ .- m.c)) + log(sum(m.q./(μ .- m.c))) - m.ε
    return isnan(val) ? typemax(val) : val
  end

  return h, (μmin, μmax)
end


function optimal(d::Burg, m::Model, μ::Real) 
  p = m.q./(μ .- m.c)
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
  check_ε(d::Hellinger, m::Model)

Returns `true` if the constraint for the ε parameter for the Hellinger distance is met
and `false` otherwise.
"""
check_ε(d::Hellinger, m::Model) = m.ε < 2 - 2*sqrt(sum(m.q[m.Imax]))

function find_mu(d::Hellinger, m::Model)
  μmin = m.cmax
  μmax = m.cmax + (2 - m.ε)*(m.cmax - m.cmin)/m.ε

  function h(μ)
    val =  2*sum(m.q./(μ .- m.c)) - (2 - m.ε)*sqrt(sum(m.q./((μ .- m.c).^2)))
    return isnan(val) ? - typemax(val) : val
  end

  return h, (μmin, μmax)
end


function optimal(d::Hellinger, m::Model, μ::Real) 
  p = m.q./(μ .- m.c).^2
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
  check_ε(d::ChiSquare, m::Model)

Returns `true` if the constraint for the ε parameter for the χ²-distance is met
and `false` otherwise.
"""
check_ε(d::ChiSquare, m::Model) = true


function find_mu(d::ChiSquare, m::Model)
  μmin = m.cmax
  μmax = Inf

  function h(μ)
    val = sum(m.q .* sqrt.(μ .- m.c))*sum(m.q./sqrt.(μ .- m.c)) - 1 - m.ε
    return isnan(val) ? - typemax(val) : val
  end

  return h, (μmin, μmax)
end


function optimal(d::ChiSquare, m::Model, μ::Real) 
  p = m.q./sqrt.(μ .- m.c)
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
  check_ε(d::ModifiedChiSquare, m::Model)

Returns `true` if the constraint for the ε parameter for the modified χ²-distance is met
and `false` otherwise.
"""
check_ε(d::ModifiedChiSquare, m::Model) = true


function find_mu(d::ModifiedChiSquare, m::Model)
  μmin = - m.cmax
  μmax = Inf

  function h(μ)
    val = sum(m.q .* max.(μ .+ m.c, 0).^2) - (1 + m.ε) * (sum(m.q .* max.(μ .+ m.c, 0)))^2
    return isnan(val) ? - typemax(val) : val
  end

  return h, (μmin, μmax)
end


function optimal(d::ModifiedChiSquare, m::Model, μ::Real) 
  p = m.q.*max.(μ .+ m.c, 0)
  return p./sum(p)
end