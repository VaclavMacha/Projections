function simplex_mod2_exact(p0::AbstractArray{<:Real},
                            q0::AbstractArray{<:Real},
                            C1::Real,
                            C2::Integer;
                            verbose::Bool = false)

    p = Convex.Variable(length(p0))
    q = Convex.Variable(length(q0))

    objective   = sumsquares(p - p0)/2 + sumsquares(q - q0)/2
    constraints = [sum(p) == sum(q),
                   p <= C1,
                   q <= sum(p)/C2,
                   p >= 0,
                   q >= 0]

    problem = Convex.minimize(objective, constraints)
    Convex.solve!(problem, ECOSSolver(verbose = verbose), verbose = verbose)

    return p.value, q.value
end


function find_λ_mod2(μ, s, C2::Integer)
    i, j, d  = 2, 1, 1
    λ, λ_old = s[1], 0
    g, g_old = -C2*μ, -C2*μ

    while g < 0
        g_old = g
        λ_old = λ

        if s[i] <= s[j] + μ
            g += d*(s[i] - λ)
            λ  = s[i]
            d += 1
            i += 1
        else
            g += d*(s[j] + μ - λ)
            λ  = s[j] + μ
            d -= 1
            j += 1
        end
    end
    return -(λ - λ_old)/(g - g_old)*g_old + λ_old
end


function find_μ_mod2(μ, s, p0, q0, C1::Real, C2::Integer)
    add_eval!()
    λ = find_λ_mod2(μ, s, C2)
    return sum(min.(max.(p0 .- λ .+ sum(max.(q0 .+ λ .- μ, 0))/C2, 0), C1)) - C2*μ
end


function simplex_mod2(p0::AbstractArray{<:Real},
                      q0::AbstractArray{<:Real},
                      C1::Real,
                      C2::Integer;
                      returnstats::Bool = false,
                      kwargs...)

    reset_stats!()
    n  = length(p0)
    λ  = 0
    μ  = n*C1/(2*C2)/100
    lb = 1e-10
    ub = n*C1/C2 + 1e-6

    if C2 >= length(q0)
        @error "No feasible solution: C2 < length(q0) needed."

        add_stat!(:μ => μ, :λ => λ, :lb => lb, :ub => ub)
        return returnstats ? (p0, q0, return_stats(), return_evals()) : (p0, q0)
    end


    if mean(partialsort(q0, 1:C2; rev = true)) + maximum(p0) <= 0
        p = zero(p0)
        q = zero(q0)
    else
        s      = vcat(.- sort(q0; rev = true), Inf)
        g_μ(μ) = find_μ_mod2(μ, s, p0, q0, C1, C2)

        μ = find_root(g_μ, μ, lb, ub ; kwargs...)
        λ = find_λ_mod2(μ, s, C2)

        δ = sum(max.(q0 .+ λ .- μ, 0))/C2
        p = @. max(min(p0 - λ + δ, C1), 0)
        q = @. max(min(q0 + λ, μ), 0)
    end


    add_stat!(:μ => μ, :λ => λ, :lb => lb, :ub => ub)
    return returnstats ? (p, q, return_stats(), return_evals()) : (p, q)
end
