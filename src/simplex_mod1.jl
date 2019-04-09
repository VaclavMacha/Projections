function simplex_mod1_exact(p0, q0, r0::Real, C1::Real, C2::Real; verbose::Bool = false)
	p = Convex.Variable(length(p0))
	q = Convex.Variable(length(q0))
	r = Convex.Variable()

	objective   = sumsquares(p - p0)/2 + sumsquares(q - q0)/2 + sumsquares(r - r0)/2
	constraints = [sum(p) == sum(q),
				   p <= C1,
				   q <= C2*r,
				   p >= 0,
				   q >= 0]

	problem = Convex.minimize(objective, constraints)
	Convex.solve!(problem, ECOSSolver(verbose = verbose), verbose = verbose)

	return p.value, q.value, r.value
end

function find_λ(μ::Real, q0::AbstractArray{<:Real}, r0::Real, C2::Real)
	return C2*r0 + C2^2*sum(max.(q0 .- μ, 0)) - μ
end

function find_μ(μ::Real, p0::AbstractArray{<:Real}, q0::AbstractArray{<:Real}, r0::Real, C1::Real, C2::Real)
	λ = find_λ(μ, q0, r0, C2)
	return sum(min.(max.(p0 .- λ, 0), C1)) - sum(min.(max.(q0 .+ λ, 0), λ + μ))
end

function simplex_mod1(p0::AbstractArray{<:Real}, q0::AbstractArray{<:Real}, r0::Real, C1::Real, C2::Real; verbose::Bool = true)

	if r0 <= - C2*sum(max.(q0 .+ maximum(p0), 0))
		p = zero(p0)
		q = zero(q0)
		r = zero(r0)
	else
		λ, μ   = 0, 0
		g_μ(μ) = find_μ(μ, p0, q0, r0, C1, C2)

		try
			μ = Roots.find_zero(g_μ, 1, Order1())
			λ + μ <= 0 && @error "Wrong solution"
		catch
			verbose && @warn "Secant method failed -> Bisection"
			lb1 = minimum(C2*r0 + C2^2*sum(q0) .- p0)/(C2^2*length(q0) + 1)
			lb2 = minimum(q0)
			lb  = min(lb1, lb2)
			ub  = maximum(q0) + C2*max(r0, 0)

			μ = Roots.find_zero(g_μ, (lb, ub), Bisection())
		end

		λ = find_λ(μ, q0, r0, C2)
		p = @. min(max(p0 - λ, 0), C1)
		q = @. min(max(q0 + λ, 0), λ + μ)
		r = (λ + μ)/C2
	end
	return p, q, r
end
