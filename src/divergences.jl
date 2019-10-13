"""
  Divergence

An abstract type covering all ϕ-divergence.
"""
abstract type Divergence end


# ----------------------------------------------------------------------------------------------------------
# Solvers
# ----------------------------------------------------------------------------------------------------------
"""
  solve_exact(d::Divergence, m::Model)

Solves the given model `m` with ϕ-divergence `d` using the Ipopt solver. 
"""
function solve_exact(d::Divergence, m::Model)

    model = JuMP.Model(JuMP.with_optimizer(Ipopt.Optimizer, print_level = 0))

    ϕ = generate(d)
    JuMP.register(model, :ϕ, 1, ϕ, autodiff = true)

    JuMP.@variable(model, p[1:length(m.q)] >= 0)

    JuMP.@objective(model, Max, m.c'*p)
    JuMP.@constraint(model, sum(p) == 1)
    JuMP.@NLconstraint(model, sum(m.q[i]*ϕ(p[i]/m.q[i]) for i in eachindex(p)) <= m.ε)

    JuMP.optimize!(model)
    return JuMP.value.(p)
end


"""
  solve(d::Divergence, m::Model)

Solves the given model `m` with ϕ-divergence `d` using our new approach. 
"""
function solve(d::Divergence, m::Model)
    p = zero(m.q)
    p[m.Imax] .= m.q[m.Imax]
    p ./= sum(m.q[m.Imax])

    ϕ = generate(d)

    if sum(m.q .* ϕ.(p ./ m.q)) <= m.ε
      return p
    else
      return optimal(d, m, findroot(d, m))
    end
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
  check_ε(d::KullbackLeibler, m::Model)

Returns `true` if the constraint for the ε parameter for the Kullback-Leibler divergence is met
and `false` otherwise.
"""
check_ε(d::KullbackLeibler, m::Model) = m.ε < -log(sum(m.q[m.Imax]))


bounds(d::KullbackLeibler, m::Model) = (0, (m.cmax - m.cmin)/m.ε)


function h(d::KullbackLeibler, m::Model, λ)
  p_hat = m.q.*exp.(m.c./λ)
  val   = p_hat' * (m.c./λ .- log(sum(p_hat)) .- m.ε)
  return isnan(val) || val == - typemax(val) ? typemax(val) : val
end


function optimal(d::KullbackLeibler, m::Model, λ::Real) 
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
  check_ε(d::Burg, m::Model)

Returns `true` if the constraint for the ε parameter for the Burg entropy is met
and `false` otherwise.
"""
check_ε(d::Burg, m::Model) = true


bounds(d::Burg, m::Model) = (m.cmax, m.cmax + (m.cmax - m.cmin)/m.ε)


function h(d::Burg, m::Model, λ)
  val = sum(m.q.*log.(λ .- m.c)) + log(sum(m.q./(λ .- m.c))) - m.ε
  return isnan(val) ? typemax(val) : val
end


function optimal(d::Burg, m::Model, λ::Real) 
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
  check_ε(d::Hellinger, m::Model)

Returns `true` if the constraint for the ε parameter for the Hellinger distance is met
and `false` otherwise.
"""
check_ε(d::Hellinger, m::Model) = m.ε < 2 - 2*sqrt(sum(m.q[m.Imax]))


bounds(d::Hellinger, m::Model) = (m.cmax, m.cmax + (2 - m.ε)*(m.cmax - m.cmin)/m.ε)


function h(d::Hellinger, m::Model, λ)
  val =  2*sum(m.q./(λ .- m.c)) - (2 - m.ε)*sqrt(sum(m.q./((λ .- m.c).^2)))
  return isnan(val) ? - typemax(val) : val
end


function optimal(d::Hellinger, m::Model, λ::Real) 
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
  check_ε(d::ChiSquare, m::Model)

Returns `true` if the constraint for the ε parameter for the χ²-distance is met
and `false` otherwise.
"""
check_ε(d::ChiSquare, m::Model) = true


bounds(d::ChiSquare, m::Model) = (m.cmax, Inf)


function h(d::ChiSquare, m::Model, λ)
  val = sum(m.q .* sqrt.(λ .- m.c))*sum(m.q./sqrt.(λ .- m.c)) - 1 - m.ε
  return isnan(val) ? - typemax(val) : val
end


function ∇h(d::ChiSquare, m::Model, λ::Real)
  λc = λ .- m.c
  return (sum(m.q ./ sqrt.(λc))^2 - sum(m.q .* sqrt.(λc))*sum(m.q ./ (λc.^(3/2))))/2
end


function optimal(d::ChiSquare, m::Model, λ::Real) 
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
  check_ε(d::ModifiedChiSquare, m::Model)

Returns `true` if the constraint for the ε parameter for the modified χ²-distance is met
and `false` otherwise.
"""
check_ε(d::ModifiedChiSquare, m::Model) = true


bounds(d::ModifiedChiSquare, m::Model) = (- m.cmax, Inf)


function h(d::ModifiedChiSquare, m::Model, λ)
  val = sum(m.q .* max.(λ .+ m.c, 0).^2) - (1 + m.ε) * (sum(m.q .* max.(λ .+ m.c, 0)))^2
  return isnan(val) ? - typemax(val) : val
end


function ∇h(d::ModifiedChiSquare, m::Model, λ::Real)
    I = findall(λ .+ m.c .>= 0) 
    return 2 * sum(m.q[I] .* (λ .+ m.c[I])) * (1 - (1 + m.ε) * sum(m.q[I]))
end


function optimal(d::ModifiedChiSquare, m::Model, λ::Real) 
  p = m.q.*max.(λ .+ m.c, 0)
  return p./sum(p)
end