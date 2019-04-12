function philpott(p0::AbstractArray{<:Real},
                  c::AbstractArray{<:Real},
                  ε::Real)

    m = length(p0)
    K = BitArray(ones(size(p0)))
    Klen = sum(K)
    p = zero(p0)
    c = - c
    j = 0
    a1, a2, a3 = 0, 0, sqrt(m)*ε


    while Klen > 1
        c_bar = mean(c[K])
        s     = sqrt(mean(c[K].^2) - c_bar^2)

        if Klen == m
            p = p0 .+ ε*(c .- c_bar)/(sqrt(m)*s)
        else
            a1 = @views sum(p0[.~K])
            a2 = @views sum(p0[.~K].^2)
            a3 = @views sqrt(Klen*(ε^2 - a2) - a1^2)

            @. p[K]  = p0[K] + (a1 + a3*(c[K] - c_bar)/s)/Klen
            @. p[~K] = 0
        end

        if minimum(p) >= 0
            return p
        else
            K2 = K .* (p .< 0)
            ε_new  = @. @views (a1^2 + (s*(Klen*p0[K2] + a1)/(c[K2] - c_bar))^2)/Klen + a2
            val, j = findmin(ε_new)
            K[findall(K2)[j]] = false
        end

        Klen = sum(K)
    end
    @. p[K] = 1
    @. p[~K] = 0
    return p
end


function philpott_optimized(p0::AbstractArray{<:Real},
                            c::AbstractArray{<:Real},
                            ε::Real)

    m = length(p0)
    K = BitArray(ones(size(p0)))
    p = zero(p0)
    c = - c
    j = 0
    a1, a2, a3 = 0, 0, sqrt(m)*ε

    for M = m:-1:1
        c_bar = mean(c[K])
        s     = sqrt(mean(abs2, c[K]) - c_bar^2)

        if M == m
            p = p0 .+ ε*(c .- c_bar)/(sqrt(m)*s)
        else
            a1 += p0[j]
            a2 += p0[j]^2
            a3 = sqrt(M*(ε^2 - a2) - a1^2)

            @. p[K]  = p0[K] + (a1 + a3*(c[K] - c_bar)/s)/M
            p[j] = 0
        end

        if minimum(p) >= 0
            return p
        else
            K2     = @views p .< 0
            ε_new  = @. (a1^2 + (s*(M*p0[K2] + a1)/(c[K2] - c_bar))^2)/M + a2
            val, j = findmin(ε_new)
            j      = findall(K2)[j]
            K[j]   = false
        end
    end
    @. p[K] = 1
    return p
end