"""
    Philpott

An empty structure representing Philpott's approach for l-2 norm.
"""
struct Philpott <: Norm end


"""
    name(d::Philpott)

Returns the full name of the Philpott's method.
"""
name(d::Philpott) = "l-2 norm (Philpott)"


"""
    normtype(d::Philpott)

Returns the type of l-2 norm.
"""
normtype(d::Philpott) = 2


"""
    solve(d::Philpott, m::Model)

Solves the given model `m` with l-2 norm using Philpott's approach. 
"""
function solve(d::Philpott, m::Model; atol::Real = 1e-10)
    n      = length(m.q)
    c_mean = Statistics.mean(m.c)
    c_std  = Statistics.stdm(m.c, c_mean; corrected = false)
    p      = m.q .+ m.ε .* (m.c .- c_mean) ./ (sqrt(n) * c_std)

    isapprox(c_std, 0, atol = atol)         && return m.q
    m.ε <= sqrt(n / (n - 1)) * minimum(m.q) && return p

    I  = collect(1:n)
    j  = 0
    a1 = 0
    a2 = 0
    a3 = sqrt(n)*m.ε

    @views for I_len in n-1:-1:1

        minimum(p[I]) >= 0 && return p

        ## find index j
        J   = I[p[I] .< 0]
        ε_p = @. (a1^2 + (c_std * (I_len * m.q[J] + a1) / (m.c[J] - c_mean))^2) / I_len + a2
        j   = J[findmin(ε_p)[2]]

        deleteat!(I, searchsortedfirst(I, j))
        p[j] = 0

        ## update p
        a1 += m.q[j]
        a2 += m.q[j]^2
        a3 = sqrt(I_len*(m.ε^2 - a2) - a1^2)

        c_mean = ((I_len + 1) * c_mean - m.c[j]) / I_len
        c_std  = Statistics.stdm(m.c[I], c_mean; corrected = false)

        @. p[I] = m.q[I] + (a1 + a3 * (m.c[I] - c_mean) / c_std) / I_len

        if isapprox(c_std, 0, atol = atol)
            @. p[I] = m.q[I] + a1 / I_len
            return p
        end
    end
    p[I] = 1
    return p
end