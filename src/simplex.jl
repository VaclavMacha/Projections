# ----------------------------------------------------------------------------------------------------------
# Simplex
# ----------------------------------------------------------------------------------------------------------
function h(m::Simplex)
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
    optimal(s::Sadda, m::Simplex)

Returns the optimal solution of the simplex model 'm' with
two linear constraints and lower and upper bounds. 
"""
function optimal(s::Sadda, m::Simplex; kwargs...)
    λ = h(m)
    
    return min.(m.ub, max.(m.lb, m.q .- λ))
end