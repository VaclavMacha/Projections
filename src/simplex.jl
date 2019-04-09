function simplex_exact(p0::AbstractArray{<:Real}; verbose::Bool = false)
    p = Convex.Variable(length(p0))

	objective   = sumsquares(p - p0)/2
	constraints = [sum(p) == 1,
				   p >= 0]

	problem = Convex.minimize(objective, constraints)
	Convex.solve!(problem, ECOSSolver(verbose = verbose), verbose = verbose)

	return p.value
end

function simplex(p0::AbstractArray{<:Real})
	μ = 0
	p = sort(p0)
	n = length(p)

	p0_min = minimum(p0)
	f 	   = sum(p0) .- n*p0_min .- 1
	f_old  = f

	if f < 0
		μ = p0_min + f/n
	else
		for i in 1:n
			f_old = f
			f -= (n - i)*(p[i + 1] - p[i])
			if f <= 0
				μ = p[i] - (p[i + 1] - p[i])/(f - f_old)*f_old
				break
			end
		end
	end
	return @. max(p0 - μ, 0)
end
