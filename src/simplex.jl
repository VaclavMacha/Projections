# ----------------------------------------------------------------------------------------------------------
# Simplex 1
# ----------------------------------------------------------------------------------------------------------
function h(m::Simplex1)
    π = sortperm(m.q .- m.lb)
    χ = sortperm(m.q .- m.ub)
    s = (m.q .- m.lb)[π]
    r = vcat((m.q .- m.ub)[χ], Inf)

    i     = 1
    j     = 2
    a_hat = 1
    λ     = r[1]
    g     = sum(m.ub) - 1
    λ_old = 0
    g_old = 0

    while g > 0
        (i <= length(s) && j <= length(r)) || break

        g_old = g
        λ_old = λ
       
        if r[j] <= s[i]
            g     -= a_hat*(r[j] - λ)
            a_hat += 1
            λ      = r[j]
            j     += 1
        else
            g     -= a_hat*(s[i] - λ)
            a_hat -= 1
            λ      = s[i]
            i     += 1
        end
    end
    return λ_old - (λ - λ_old)/(g - g_old)*g_old
end


"""
    optimal(s::Sadda, m::Simplex1)

Returns the optimal solution of the simplex model 'm' with
two linear constraints and lower and upper bounds. 
"""
function optimal(s::Sadda, m::Simplex1; kwargs...)
    λ = h(m)
    
    return min.(m.ub, max.(m.lb, m.q .- λ))
end


# ----------------------------------------------------------------------------------------------------------
# Simplex 2
# ----------------------------------------------------------------------------------------------------------
function g(m::Simplex2, μ::Real)
    l_hat = m.lb ./ m.a
    u_hat = m.ub ./ m.a
    q_hat = (m.q .- μ .* m.b) ./ m.a

    π = sortperm(q_hat .- l_hat)
    χ = sortperm(q_hat .- u_hat)
    s = (q_hat .- l_hat)[π]
    r = vcat((q_hat .- u_hat)[χ], Inf)

    i     = 1
    j     = 2
    a_hat = m.a[χ[1]].^2
    λ     = r[1]
    g     = sum(m.a.^2 .* u_hat) - m.C1
    λ_old = 0
    g_old = 0

    while g > 0
        (i <= length(s) && j <= length(r)) || break 
 
        g_old = g
        λ_old = λ
        
        if r[j] <= s[i]
            g     -= a_hat*(r[j] - λ)
            a_hat += m.a[χ[j]]^2
            λ      = r[j]
            j     += 1
        else
            g     -= a_hat*(s[i] - λ)
            a_hat -= m.a[π[i]]^2
            λ      = s[i]
            i     += 1
        end
    end
    return λ_old - (λ - λ_old)/(g - g_old)*g_old
end


function h(m::Simplex2, μ::Real; λ::Real = g(m, μ))
    return sum(m.b .* min.(m.ub, max.(m.lb, m.q .- λ .* m.a .- μ .* m.b))) - m.C2
end


"""
    bounds(m::Simplex2)

Returns lower and upper bound for finding the root of the function `h` using the bisection method. 
"""
bounds(m::Simplex2) = (-1e10, 1e10)


"""
    optimal(s::Sadda, m::Simplex2)

Returns the optimal solution of the simplex model 'm' with
two linear constraints and lower and upper bounds. 
"""
function optimal(s::Sadda, m::Simplex2; kwargs...)
    μ = bisection(m; kwargs...)
    λ = g(m, μ)

    return min.(m.ub, max.(m.lb, m.q .- λ .* m.a .- μ .* m.b))
end
