function simplex_mod1_exact(p0::AbstractArray{<:Real}, q0::AbstractArray{<:Real}, r0::Real, C1::Real, C2::Real)
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
	Convex.solve!(problem, SCSSolver(verbose = 0))

	return p.value, q.value, r.value
end

function find_λ(μ::Real, q0::AbstractArray{<:Real}, r0::Real, C2::Real)
	return C2*r0 + C2^2*sum(max.(q0 .- μ, 0)) - μ
end

function find_μ(μ::Real, p0::AbstractArray{<:Real}, q0::AbstractArray{<:Real}, r0::Real, C1::Real, C2::Real)
	λ = find_λ(μ, q0, r0, C2)
	return sum(min.(max.(p0 .- λ, 0), C1)) - sum(min.(max.(q0 .+ λ, 0), λ + μ))
end

function simplex_mod1(p0::AbstractArray{<:Real}, q0::AbstractArray{<:Real}, r0::Real, C1::Real, C2::Real)

	if r0 <= - C2*sum(max.(q0 .+ maximum(p0), 0))
		p = zero(p0)
		q = zero(q0)
		r = zero(r0)
	else
		λ, μ, lb = 0, 0, 1e-6
		g_μ(μ)   = find_μ(μ, p0, q0, r0, C1, C2)

		try
			μ = Roots.find_zero(g_μ, 1, Order1())
			λ + μ <= lb && @error "Wrong solution"
		catch
			@warn "Secant method failed -> Bisection"
			μ = Roots.find_zero(g_μ, (lb, maximum(q0) + C2*r0), Bisection())
		end

		λ = find_λ(μ, q0, r0, C2)
		p = @. min(max(p0 - λ, 0), C1)
		q = @. min(max(q0 + λ, 0), λ + μ)
		r = (λ + μ)/C2
	end
	return p, q, r
end
