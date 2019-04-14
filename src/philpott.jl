function philpott(p0::AbstractArray{<:Real},
                  c::AbstractArray{<:Real},
                  ε::Real;
                  atol::Real = 1e-10)

    m      = length(p0)
    c      = - c
    p      = zero(p0)
    c_mean = mean(c)
    c_std  = sqrt(mean(abs2, c .- c_mean))

    isapprox(c_std, 0, atol = atol) && return p0

    I, I_len, j = BitArray(ones(size(p0))), m, 0
    a1, a2, a3  = 0, 0, sqrt(m)*ε

    while I_len > 1
        c_mean = mean(c[I])
        c_std  = sqrt(mean(abs2, c[I] .- c_mean))

        if isapprox(c_std, 0, atol = atol)
            a1       = @views sum(p0[.~I])
            @. p[I]  = p0[I] + a1/I_len
            @. p[~I] = 0
            return p
        end

        ## update p
        if I_len == m
            p = p0 .+ ε*(c .- c_mean)/(sqrt(m)*c_std)

            ε <= sqrt(m/(m-1))*minimum(p0) && return p
        else
            a1 = @views sum(p0[.~I])
            a2 = @views sum(p0[.~I].^2)
            a3 = @views sqrt(I_len*(ε^2 - a2) - a1^2)

            @. p[I]  = p0[I] + (a1 + a3*(c[I] - c_mean)/c_std)/I_len
            @. p[~I] = 0
        end

        ## find index j
        if minimum(p) >= 0
            return p
        else
            J   = I .* (p .< 0)
            ε_p = @. (a1^2 + (c_std*(I_len*p0[J] + a1)/(c[J] - c_mean))^2)/I_len + a2
            j   = findmin(ε_p)[2]

            I[findall(J)[j]] = false
            I_len = sum(I)
        end
    end
    @. p[I] = 1
    @. p[~I] = 0
    return p
end


function philpott_optimized(p0::AbstractArray{<:Real},
                            c::AbstractArray{<:Real},
                            ε::Real;
                            atol::Real = 1e-10,
                            kwargs...)

    c      = - c
    m      = length(p0)
    c_mean = mean(c)
    c_std  = stdm(c, c_mean; corrected = false)
    p      = p0 .+ ε*(c .- c_mean)/(sqrt(m)*c_std)

    isapprox(c_std, 0, atol = atol) && return p0
     ε <= sqrt(m/(m-1))*minimum(p0) && return p

    I, j       = collect(1:m), 0
    a1, a2, a3 = 0, 0, sqrt(m)*ε

    @views for I_len = m-1:-1:1

        minimum(p[I]) >= 0 && return p

        ## find index j
        J   = I[p[I] .< 0]
        ε_p = @. (a1^2 + (c_std*(I_len*p0[J] + a1)/(c[J] - c_mean))^2)/I_len + a2

        j = J[findmin(ε_p)[2]]
        deleteat!(I, searchsortedfirst(I, j))
        p[j] = 0

        ## update p
        a1 += p0[j]
        a2 += p0[j]^2
        a3 = sqrt(I_len*(ε^2 - a2) - a1^2)

        c_mean = ((I_len + 1)*c_mean - c[j])/I_len
        c_std  = stdm(c[I], c_mean; corrected = false)

        @. p[I] = p0[I] + (a1 + a3*(c[I] - c_mean)/c_std)/I_len

        if isapprox(c_std, 0, atol = atol)
            @. p[I] = p0[I] + a1/I_len
            return p
        end
    end
    p[I] = 1
    return p
end