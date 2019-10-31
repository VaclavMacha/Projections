"""
    newton(d::Union{ChiSquare, ModifiedChiSquare}, m::DRO; maxiter::Integer = 1000, atol::Real = 1e-6)

A simple bisection method for finding the root of the function `h` for the DRO problem with
χ²-distance or modified χ²-distance.
"""
function newton(d::Union{ChiSquare, ModifiedChiSquare}, m::DRO; maxiter::Integer = 1000, atol::Real = 1e-6)
    stats.optimizer  = "newton"
    f(λ)  = h(d, m, λ)
    ∇f(λ) = ∇h(d, m, λ)

    λ     = initial(d,m)    
    f_val = f(λ)
    add_eval()

    f_val == 0 && return λ

    for k in 1:maxiter
        abs(f_val) > atol || return λ

        λ    -= f_val/∇f(λ)
        f_val = f(λ)

        add_eval()
    end
    @warn "Newton method for the DRO model with $(name(d)) reached maximal
           number of iteration. The solution may not be accurate: h(μ) = $(f(λ))" 
    return λ
end


"""
    newton(d::Ltwo, m::DRO; maxiter::Integer = 1000, atol::Real = 1e-6)

A simple bisection method for finding the root of the function `h` for the DRO problem with
l-2 norm.
"""
function newton(d::Ltwo, m::DRO; maxiter::Integer = 1000, atol::Real = 1e-6)
    stats.optimizer  = "newton"
    f(μ, λ)  = h(d, m, μ; λ = λ)
    ∇f(μ, λ) = ∇h(d, m, μ; λ = λ)

    μ     = initial(d,m)
    λ     = g(d, m, μ)
    f_val = f(μ, λ)
    add_eval()

    f_val == 0 && return μ

    for k in 1:maxiter
        abs(f_val) > atol || return μ

        μ    -= f_val/∇f(μ, λ)
        λ     = g(d, m, μ)
        f_val = f(μ, λ)
        add_eval()
    end
    @warn "Newton method for the DRO model with $(name(d)) reached maximal
           number of iteration. The solution may not be accurate: h(μ) = $(f(μ))"
    return μ
end


"""
    bisection(d::Union{KullbackLeibler, Burg, Hellinger}, m::DRO; kwargs...)

A simple bisection method for finding the root of the function `h` for the DRO problem with
Kullback-Leibler divergence, Burg entropy or Hellinger distance.
"""
function bisection(d::Union{KullbackLeibler, Burg, Hellinger}, m::DRO; kwargs...)
    f(μ) = h(d, m, μ)
    a, b = bounds(d,m)

    return bisection(f, a, b; kwargs...)
end


"""
    bisection(m::Simplex; kwargs...)

A simple bisection method for finding the root of the function `h` for the Simplex problems.
"""
function bisection(m::Simplex; kwargs...)
    f(μ) = h(m, μ)
    a, b = bounds(m)

    return bisection(f, a, b; kwargs...)
end


"""
    bisection(f::Function, a::Real, b::Real; maxiter::Integer = 1000, atol::Real = 1e-8)

A simple bisection method for finding the root of the function `f`.
"""
function bisection(f::Function, a::Real, b::Real; maxiter::Integer = 1000, atol::Real = 1e-8)
    stats.optimizer  = "bisection"
    local c

    f_a = f(a)
    f_b = f(b)
    add_eval(2)
    @assert f_a*f_b <= 0 "The interval [a,b] is not a bracketing interval."
    
    f_a == 0 && return a
    f_b == 0 && return b

    for k in 1:maxiter
        abs(b - a) > atol || return c

        c   = Statistics.middle(a, b)
        f_c = f(c)
        add_eval()

        f_c == 0 && return c

        if f_a*f_c > 0
            a, f_a = c, f_c
        else
            b = c
        end
    end
   @warn "Bisection method reached maximal number of iteration.
          The solution may not be accurate: h(μ) = $(f(c))"
    return c
end