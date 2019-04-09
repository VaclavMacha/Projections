function minimize_linear_on_simplex_exact(p0, c, ε, k; verbose::Bool = false)
	p = Convex.Variable(length(p0))

	objective   = c'*p
	constraints = [sum(p) == 1,
				   p >= 0,
				   norm(p - p0, k) <= ε]

	problem = Convex.minimize(objective, constraints)
	Convex.solve!(problem, ECOSSolver(verbose = verbose), verbose = verbose)
	return vec(p.value)
end


## lInf norm
function minimize_linear_on_simplex_lInf(p0, c, ε)

	if !isapprox(sum(p0), 1, atol = 1e-8) || minimum(p0) < 0
		@error string("p0 is not a probability distribution: ∑p0 = ", sum(p0), ", min(p0) = ", minimum(p0))
		return p0
	end

	perm = sortperm(c)
	c = c[perm]
	p = p0[perm]
	i = 1
	j = length(c)


	δ_down = 0
	while i < j
		δ1    = min(1 - p[i], ε)
		δ2    = 0
		p[i] += δ1

		while δ2 < δ1

			if p[j] >= δ1 - δ2
				if δ1 - δ2 <= ε - δ_down
					p[j]   -= δ1 - δ2
					δ_down += δ1 - δ2
					break
				else
					δ2    += ε - δ_down
					p[j]  -= ε - δ_down
					δ_down = 0
					j -= 1
				end
			else
				δ2     += min(p[j], ε - δ_down)
				p[j]   -= min(p[j], ε - δ_down)
				δ_down = 0
				j      -= 1
			end

			if i == j
				p[i] += - δ1 + δ2
				break
			end

		end
		i += 1
	end
	return p[invperm(perm)]
end

## l1 norm
function minimize_linear_on_simplex_l1(p0, c, ε)

	if !isapprox(sum(p0), 1, atol = 1e-8) || minimum(p0) < 0
		@error string("p0 is not a probability distribution: ∑p0 = ", sum(p0), ", min(p0) = ", minimum(p0))
		return p0
	end

	perm = sortperm(c)
	c = c[perm]
	p = p0[perm]
	i = 1
	j = length(c)
	δ_up = 0

	while i < j && δ_up < ε/2
		δ1   = min(1 - p[i], ε/2-δ_up)
		δ_up += δ1
		δ2   = 0
		p[i] += δ1

		while δ2 < δ1
			if p[j] >= δ1 - δ2
				p[j] += - δ1 + δ2
				break
			else
				δ2 = δ2 + p[j]
				p[j] = 0
				j -= 1
				if i == j
					p[i] += - δ1 + δ2
					break
				end
			end
		end
		i += 1
	end
	return p[invperm(perm)]
end


## l2 norm
function fλ(μ, p0, c)
	n = length(p0)
	s = sort(μ*p0 - c)
	g = n*s[1] + sum(c)
	g_old = 0
	λ = 0

	if g > 0
		λ = - mean(c)
	else
		for i in 1:n-1
			g_old = g
			g += (n - i)*(s[i + 1] - s[i])
			if g >= 0
				λ = s[i] - (s[i + 1] - s[i])/(g - g_old)*g_old
				break
			end
			# i == n - 1 && @error "khbi"
		end
	end
	return λ
end

function fμ(μ, p0, c, ε)
	λ = fλ(μ, p0, c)
	return sum((min.(c .+ λ, μ.*p0)).^2) - ε^2*μ^2
end


function minimize_linear_on_simplex_l2(p0, c, ε)

	if !isapprox(sum(p0), 1, atol = 1e-8) || minimum(p0) < 0
		@error string("p0 is not a probability distribution: ∑p0 = ", sum(p0), ", min(p0) = ", minimum(p0))
		return p0
	end

	if iszero(ε)
		return p0
	end

	c_min = minimum(c)
	i_min = c_min .== c
	p     = zero(p0)

	if sum(i_min) == 1
		p[i_min] .= 1
	else
		p[i_min] .= simplex(p0[i_min])
	end

	if norm(p - p0) > ε
		λ = 0
		μ = 0

		gμ(μ) = fμ(μ, p0, c, ε)
		try
			μ = Roots.find_zero(gμ, 100, Order1())
			μ <= 1e-6 && @error("Wrong solution")
		catch
			@warn "Secant method failed -> Bisection"
			μ = Roots.find_zero(gμ, (1e-6, 1e10), Bisection())
		end

		λ = fλ(μ, p0, c)
		p = @. max(p0 - (c + λ)/μ, 0)
	end
	return p
end
