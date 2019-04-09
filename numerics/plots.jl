## Simplex mod2
function find_μ_simplex_mod1(l::Real;
							 p0_dist = Normal(0,1),
							 q0_dist = Normal(0,1),
							 r0_dist = Uniform(0,1),
							 C1::Real = 0.1,
							 C2::Real = 0.8,
							 seed::Integer = 1234,
							 kwargs...)
	Random.seed!(seed);

	n  = ceil(Int64, 10^l)
	n1 = ceil(Int64, n/3)
	n2 = n - n1 - 1

	p0 = rand(p0_dist, n1)
	q0 = rand(q0_dist, n2)
	r0 = rand(r0_dist)

	g_μ(μ) = Projections.find_μ(μ, p0, q0, r0, C1, C2)

	return g_μ
end


## Simplex mod2
function find_μ_simplex_mod2(l::Real;
							 p0_dist = Normal(0,1),
							 q0_dist = Normal(0,1),
							 C1::Real = 0.1,
							 C2::Integer = 0,
							 seed::Integer = 1234,
							 kwargs...)
	Random.seed!(seed);

	n  = ceil(Int64, 10^l)
	n1 = ceil(Int64, n/3)
	n2 = n - n1 - 1

	p0 = rand(p0_dist, n1)
	q0 = rand(q0_dist, n2)
	if iszero(C2)
		C2 = ceil(Int64, n2/10)
	end

	g_μ(μ) = Projections.find_μ(μ, s, p0, q0, C1, C2)

	return g_μ
end


## Minimize linear_on simplex l2
function find_μ_minimize_linear_on_simplex(l::Real;
										   c_dist  = Normal(0,1),
										   p0_dist = Uniform(0,1),
										   ε::Real = 0.1,
										   seed::Integer = 1234,
			  							   kwargs...)
	Random.seed!(seed);

	n   = ceil(Int64, 10^l)
	c   = rand(c_dist, n)
	p0  = rand(p0_dist, n)
	p0 /= sum(p0)

	g_μ(μ) = Projections.find_μ(μ, p0, c, ε)

	return g_μ
end
