function test_lower_bound(x::Real, lb::Real; var_name::String  = "x", lb_name::String = "x_lb", atol::Real = 1e-5)
	@testset "$lb_name ≦ $var_name" begin
		@test lb - atol <= x
	end
end

function test_lower_bound(x::AbstractArray{<:Real}, lb::Real; var_name::String  = "x", lb_name::String = "x_lb", atol::Real = 1e-5)
	@testset "$lb_name ≦ $var_name[$i]" for i in 1:length(x)
		@test lb - atol <= x[i]
	end
end

function test_upper_bound(x::Real, ub::Real; var_name::String  = "x", ub_name::String = "x_ub", atol::Real = 1e-5)
	@testset "$var_name ≦ $ub_name" begin
		@test x <= ub + atol
	end
end

function test_upper_bound(x::AbstractArray{<:Real}, ub::Real; var_name::String  = "x", ub_name::String = "x_lb", atol::Real = 1e-5)
	@testset "$var_name[$i] ≦ $ub_name" for i in 1:length(x)
		@test x[i] <= ub + atol
	end
end

function test_equality(x::Real, y::Real; x_name::String = "x", y_name::String = "y", atol::Real = 1e-5)
	@testset "$x_name = $y_name" begin
		@test isapprox(x, y; atol = atol)
	end
end

function test_error(err::Real; err_name::String = "relative error", atol::Real = 1e-5)
	@testset "$err_name ≈ 0" begin
		@test isapprox(err, 0; atol = atol)
	end
end

function test_error(err::AbstractArray{<:Real}; err_name::String = "relative error", atol::Real = 1e-5)
	@testset "$err_name[$i] ≈ 0" for i in 1:length(err)
		@test isapprox(err[i], 0; atol = atol)
	end
end

function test_error(x, y)
	abs_error = abs(x - y)
	max_abs   = max(abs(x), abs(y))
	rel_error = iszero(max_abs) ? 0 : abs_error/max_abs
	return min(abs_error, rel_error)
end

function test_projection_simplex(p0::AbstractArray{<:Real})
	p_opt = Projections.simplex_exact(p0)
	p     = Projections.simplex(p0)

	@testset "Test - simplex projection" begin
		# test_equality(sum(p), 1; x_name = "∑p", y_name = "1")
		# test_lower_bound(p, 0; var_name = "p", lb_name = "0")
		test_error(test_error.(p, p_opt); err_name = "err(p - p_opt)")
	end;
	return nothing
end

function test_projection_simplex_mod1(p0::AbstractArray{<:Real}, q0::AbstractArray{<:Real}, r0::Real, C1::Real, C2::Real)
	pe, qe, re = Projections.simplex_mod1_exact(p0, q0, r0, C1, C2)
	p, q, r    = Projections.simplex_mod1(p0, q0, r0, C1, C2)

	@testset "Test - simplex mod1 projection" begin
		# test_equality(sum(p), sum(q); x_name = "∑p", y_name = "∑q")
		# test_lower_bound(p, 0; var_name = "p", lb_name = "0")
		# test_upper_bound(p, C1; var_name = "p", ub_name = "C1")
		# test_lower_bound(q, 0; var_name = "q", lb_name = "0")
		# test_upper_bound(q, C2*r; var_name = "q", ub_name = "C2⋅r")

		test_error(test_error.(p, pe); err_name = "err(p - p_opt)")
		test_error(test_error.(q, qe); err_name = "err(q - q_opt)")
		test_error(test_error.(r, re); err_name = "err(r - r_opt)")
	end;
	return nothing
end

function test_projection_simplex_mod2(p0::AbstractArray{<:Real}, q0::AbstractArray{<:Real}, C1::Real, C2::Integer)
	pe, qe = Projections.simplex_mod2_exact(p0, q0, C1, C2)
	p, q    = Projections.simplex_mod2(p0, q0, C1, C2)

	@testset "Test - simplex mod2 projection" begin
		# test_equality(sum(p), sum(q); x_name = "∑p", y_name = "∑q")
		# test_lower_bound(p, 0; var_name = "p", lb_name = "0")
		# test_upper_bound(p, C1; var_name = "p", ub_name = "C1")
		# test_lower_bound(q, 0; var_name = "q", lb_name = "0")
		# test_upper_bound(q, sum(p)/C2; var_name = "q", ub_name = "∑p/C2")

		test_error(test_error.(p, pe); err_name = "err(p - p_opt)")
		test_error(test_error.(q, qe); err_name = "err(q - q_opt)")
	end;
	return nothing
end

function test_minimize_linear_on_simplex(p0::AbstractArray{<:Real}, c::AbstractArray{<:Real}, ε::Real, k::Real)
	pe = Projections.minimize_linear_on_simplex_exact(p0, c, ε, k)
	if k == Inf
		p = Projections.minimize_linear_on_simplex_lInf(p0, c, ε)
	elseif k == 1
		p = Projections.minimize_linear_on_simplex_l1(p0, c, ε)
	elseif k == 2
		p = Projections.minimize_linear_on_simplex_l2(p0, c, ε)
	else
		@error "wrong k value, k ∈ {1, 2, Inf}"
	end

	@testset "Test - minimize linear on simplex (l_$k norm)" begin
		# test_equality(sum(p), 1; x_name = "∑p", y_name = "1")
		# test_lower_bound(p, 0; var_name = "p", lb_name = "0")
		# test_upper_bound(norm(p .- p0, k), ε; var_name = "∥p - p0∥", ub_name = "ε")

		test_error(test_error.(p, pe); err_name = "err(p - p_opt)")
	end;
	return nothing
end

using Revise
using Projections, Test, Distributions, LinearAlgebra
c  = rand(Normal(0.0, 1.0), 100)
p0 = rand(100)
p0 /= sum(p0)
ε  = 0.1

q0 = rand(10)
r0 = rand()

C1 = 2
C2 = 0.7
C2_int = 9


@time @testset "All tests" begin
	test_minimize_linear_on_simplex(p0, c, ε, Inf)
	test_minimize_linear_on_simplex(p0, c, ε, 1)
	test_minimize_linear_on_simplex(p0, c, ε, 2)

	test_projection_simplex(p0)
	test_projection_simplex_mod1(p0, q0, r0, C1, C2)
	test_projection_simplex_mod2(p0, q0, C1, C2_int)
end;


## Tests
## Simplex
c_dists  = [Uniform(0,1), Normal(0,1)]
p0_dists = [Uniform(0,1)]
εs       = vcat(0.01:0.01:0.1, 0.2, 0.5, 1)
n 		 = 100

@testset "minimize on linear simplex" begin
	for (c_dist, p0_dist, ε) in Iterators.product(c_dists, p0_dists, εs)
		c   = rand(c_dist, n)
		p0  = rand(p0_dist, n)
		p0 /= sum(p0)

		test_minimize_linear_on_simplex(p0, c, ε, Inf)
		test_minimize_linear_on_simplex(p0, c, ε, 1)
		test_minimize_linear_on_simplex(p0, c, ε, 2)
	end
	# for ε in εs
	# 	c   = max.(min.(rand(Uniform(0,1), n), 0.8), 0.2)
	# 	p0  = rand(p0_dist, n)
	# 	p0 /= sum(p0)
	#
	# 	test_minimize_linear_on_simplex(p0, c, ε, Inf)
	# 	test_minimize_linear_on_simplex(p0, c, ε, 1)
	# 	test_minimize_linear_on_simplex(p0, c, ε, 2)
	# end
end
n = 10
c  = rand(Uniform(0,1), n)
p0 = rand(Uniform(0,1), n)
p0 /= sum(p0)
ε  = 0
test_minimize_linear_on_simplex(p0, c, ε, Inf)

@testset "minimize on linear simplex" begin
	for (c_dist, p0_dist, ε) in Iterators.product(c_dists, p0_dists, εs)
		c   = rand(c_dist, n)
		p0  = rand(p0_dist, n)
		p0 /= sum(p0)

		test_minimize_linear_on_simplex(p0, c, ε, Inf)
		test_minimize_linear_on_simplex(p0, c, ε, 1)
		test_minimize_linear_on_simplex(p0, c, ε, 2)
	end
end
#
#
# function foo()
# 	if true
# 		@error "a"
# 		return nothing
# 	end
# 	println("a")
# 	return 1
# end
