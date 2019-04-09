## Simplex
function benchmark_simplex(l::AbstractArray{<:Real}; kwargs...)
	out = reduce(vcat, benchmark_simplex.(l; kwargs...))
	return out[:,1], out[:,2]
end


function benchmark_simplex(l::Real;
						   p0_dist = Normal(0,1),
						   max_rep::Integer = 10,
						   seed::Integer = 1234,
						   kwargs...)
    Random.seed!(seed);

	n  = ceil(Int64, 10^l)
	p0 = rand(p0_dist, n)

	t = map(1:max_rep) do rep
		val, t = @timed Projections.simplex(p0)
		return t
	end |> rows -> reduce(vcat, rows)
	return [mean(t) std(t)]
end


## Simplex mod1
function benchmark_simplex_mod1(l::AbstractArray{<:Real}; kwargs...)
	out = reduce(vcat, benchmark_simplex_mod1.(l; kwargs...))
	return out[:,1], out[:,3], out[:,2]
end


function benchmark_simplex_mod1(l::Real;
								p0_dist = Normal(0,1),
								q0_dist = Normal(0,1),
								r0_dist = Uniform(0,1),
								C1::Real = 0.1,
								C2::Real = 0.8,
								max_rep::Integer = 10,
								seed::Integer = 1234,
   							    kwargs...)
   	Random.seed!(seed);

	n  = ceil(Int64, 10^l)
	n1 = ceil(Int64, n/3)
	n2 = n - n1 - 1

	p0 = rand(p0_dist, n1)
	q0 = rand(q0_dist, n2)
	r0 = rand(r0_dist)

	t = map(1:max_rep) do rep
		val, t = @timed Projections.simplex_mod1(p0, q0, r0, C1, C2)
		return [t val[end]]
	end |> rows -> reduce(vcat, rows)
	return [mean(t; dims = 1) std(t[:,1])]
end


## Simplex mod2
function benchmark_simplex_mod2(l::AbstractArray{<:Real}; kwargs...)
	out = reduce(vcat, benchmark_simplex_mod2.(l; kwargs...))
	return out[:,1], out[:,3], out[:,2]
end


function benchmark_simplex_mod2(l::Real;
								p0_dist = Normal(0,1),
								q0_dist = Normal(0,1),
								C1::Real = 0.1,
								C2::Integer = 0,
								max_rep::Integer = 10,
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

	t = map(1:max_rep) do rep
		val, t = @timed Projections.simplex_mod2(p0, q0, C1, C2)
		return [t val[end]]
	end |> rows -> reduce(vcat, rows)
	return [mean(t; dims = 1) std(t[:,1])]
end


## Minimize linear_on simplex
function benchmark_minimize_linear_on_simplex(l::AbstractArray{<:Real}, k::Real; kwargs...)
	out = reduce(vcat, benchmark_minimize_linear_on_simplex.(l, k; kwargs...))
	if size(out, 2) == 2
		return out[:,1], out[:,2]
	else
		return out[:,1], out[:,3], out[:,2]
	end
end


function benchmark_minimize_linear_on_simplex(l::Real,
											  k::Real;
											  c_dist  = Normal(0,1),
											  p0_dist = Uniform(0,1),
											  ε::Real = 0.1,
											  max_rep::Integer = 10,
											  seed::Integer = 1234,
				 							  kwargs...)
	Random.seed!(seed);

	n   = ceil(Int64, 10^l)
	c   = rand(c_dist, n)
	p0  = rand(p0_dist, n)
	p0 /= sum(p0)

	t = map(1:max_rep) do rep
		if k == Inf
			val, t = @timed Projections.minimize_linear_on_simplex_lInf(p0, c, ε)
			return t
		elseif k == 1
			val, t = @timed Projections.minimize_linear_on_simplex_l1(p0, c, ε)
			return t
		elseif k == 2
			val, t = @timed Projections.minimize_linear_on_simplex_l2(p0, c, ε)
			return [t val[end]]
		else
			@error "k ∉ {1, 2, Inf}"
			return NaN
		end
	end |> rows -> reduce(vcat, rows)
	return [mean(t; dims = 1) std(t[:,1])]
end
