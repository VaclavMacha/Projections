function simplex_mod4_exact(p0::AbstractArray{<:Real},
                            q0::AbstractArray{<:Real},
                            C::Integer;
                            verbose::Bool = false)

    p = Convex.Variable(length(p0))
    q = Convex.Variable(length(q0))

    objective   = sumsquares(p - p0)/2 + sumsquares(q - q0)/2
    constraints = [sum(p) == sum(q),
                   q <= sum(p)/C,
                   p >= 0,
                   q >= 0]

    problem = Convex.minimize(objective, constraints)
       Convex.solve!(problem, ECOSSolver(verbose = verbose), verbose = verbose)

       return p.value, q.value
end


function find_μ_mod4(μ, s, p0, q0, C::Integer)
    update_stats!()
    λ = find_λ_mod2(μ, s, C)
    return sum(max.(p0 .- λ .+ sum(max.(q0 .+ λ .- μ, 0))/C, 0)) - C*μ
end


function simplex_mod4(p0::AbstractArray{<:Real},
                      q0::AbstractArray{<:Real},
                      C::Integer;
                      kwargs...)

    if C >= length(q0)
        @error "No feasible solution: C < length(q0) needed."
        return p0, q0
    end

    if mean(partialsort(q0, 1:C; rev = true)) + maximum(p0) <= 0
        p = zero(p0)
        q = zero(q0)
    else
        s  = vcat(.- sort(q0; rev = true), Inf)
        n  = length(p0)
        λ  = 0
        μ  = 10
        lb = 1e-10
        ub = n*(maximum(p0) + maximum(q0))/C

        g_μ(μ) = find_μ_mod4(μ, s, p0, q0, C)

        μ = find_root(g_μ, μ, lb, ub; kwargs...)
        λ = find_λ_mod2(μ, s, C)

        δ = sum(max.(q0 .+ λ .- μ, 0))/C
        p = @. max(p0 - λ + δ, 0)
        q = @. max(min(q0 + λ, μ), 0)
    end
    return p, q, return_stats()
end
