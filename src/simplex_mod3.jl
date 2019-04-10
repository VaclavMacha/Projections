function simplex_mod3_exact(p0::AbstractArray{<:Real},
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


function find_λ_mod3(λ::Real, p0, q0)
	update_stats!()
	sum(max.(p0 .- λ, 0)) - sum(max.(q0 .+ λ, 0))
end


function simplex_mod3(p0::AbstractArray{<:Real},
					  q0::AbstractArray{<:Real},
					  r0::Real;
					  verbose::Bool = false)

	if -maximum(q0) > maximum(p0)
		p = zero(p0)
		q = zero(q0)
		r = zero(r0)
	else
		λ      = 0
		g_λ(λ) = find_λ_mod3(λ, p0, q0)

		try
			reset_stats!()
			λ = Roots.secant_method(g_λ, (maximum(p0) - maximum(q0))/2)

			if isnan(μ)
				verbose && @warn("Secant method failed.")
				error("Secant method failed")
			end
		catch
			update_key!(:bisection)
			verbose && @warn "Secant method failed -> Bisection method will be used."

			λ = Roots.bisection(g_λ, -maximum(q0), maximum(p0))
		end

		p = @. max(p0 - λ, 0)
		q = @. max(q0 + λ, 0)
		r = max(r0, 1e-4)
	end
	return p, q, r, get_stats()
end
