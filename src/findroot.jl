function findroot(d::Union{KullbackLeibler, Burg, Hellinger}, m::Model)
    return Roots.bisection(λ -> h(d, m, λ), bounds(d,m)...)
end


function findroot(d::Union{ChiSquare, ModifiedChiSquare, Ltwo}, m::Model)
    return Newton(d, m)
end


function initialization(d::Union{ModifiedChiSquare, Ltwo}, m::Model)
    f(μ)  = h(d, m, μ)

    μ     = 10 + bounds(d,m)[1]
    f_val = f(μ)

    while f_val >= 0
        f == 0 && return μ

        μ    *= 10
        f_val = f(μ)
    end
    return μ
end


function initialization(d::ChiSquare, m::Model)
    f(μ)  = h(d, m, μ)

    μ_min = bounds(d,m)[1]
    q     = 0.01
    f_val = f(μ_min + q)

    while f_val <= 0
        f == 0 && return μ_min + q

        q    /= 10
        f_val = f(μ_min + q)
    end
    return μ_min + q
end


function Newton(d::Union{ChiSquare, ModifiedChiSquare}, m::Model; maxiter::Integer = 1000, atol::Real = 1e-10)
    f(λ)  = h(d, m, λ)
    ∇f(λ) = ∇h(d, m, λ)

    λ     = initialization(d,m)
    f_val = f(λ)

    f_val == 0 && return λ

    for k in 1:maxiter
        abs(f_val) > atol || break

    λ    -= f_val/∇f(λ)
        f_val = f(λ)
    end
    return λ
end


function Newton(d::Ltwo, m::Model; maxiter::Integer = 1000, atol::Real = 1e-10)
    f(μ, λ)  = h(d, m, μ; λ = λ)
    ∇f(μ, λ) = ∇h(d, m, μ; λ = λ)

    μ     = initialization(d,m)
    λ     = g(d, m, μ)
    f_val = f(μ, λ)

    f_val == 0 && return μ

    for k in 1:maxiter
        abs(f_val) > atol || break

        μ    -= f_val/∇f(μ, λ)
        λ     = g(d, m, μ)
        f_val = f(μ, λ)
    end
    return μ
end