function solve_AATP3_exact(p0::AbstractArray{<:Real},
                           q0::AbstractArray{<:Real},
                           r0::Real;
                           verbose::Bool = false)

    p = Convex.Variable(length(p0))
    q = Convex.Variable(length(q0))
    r = Convex.Variable()

    objective   = sumsquares(p - p0)/2 + sumsquares(q - q0)/2 + sumsquares(r - r0)/2
    constraints = [sum(p) == sum(q),
                   p >= 0,
                   q >= 0,
                   r >= 0]

    problem = Convex.minimize(objective, constraints)
       Convex.solve!(problem, ECOSSolver(verbose = verbose), verbose = verbose)

       return p.value, q.value, r.value
end


function find_λ_AATP3(λ::Real, p0, q0)
    add_eval!()
    sum(max.(p0 .- λ, 0)) - sum(max.(q0 .+ λ, 0))
end


function solve_AATP3(p0::AbstractArray{<:Real},
                     q0::AbstractArray{<:Real},
                     r0::Real;
                     returnstats::Bool = false,
                     kwargs...)

    reset_stats!()
    λ  = (maximum(p0) - maximum(q0))/2
    lb = -maximum(q0)
    ub = maximum(p0)


    if -maximum(q0) > maximum(p0)
        p = zero(p0)
        q = zero(q0)
        r = zero(r0)
    else
        g_λ(λ) = find_λ_AATP3(λ, p0, q0)

        λ = find_root(g_λ, λ, lb, ub; kwargs...)
        p = @. max(p0 - λ, 0)
        q = @. max(q0 + λ, 0)
        r = max(r0, 1e-4)
    end


    add_stat!(:λ => λ, :lb => lb, :ub => ub)
    return returnstats ? (p, q, r, return_stats(), return_evals()) : (p, q, r)
end
