L(p, p0, q = 0, q0 = 0, r = 0, r0 = 0) = norm(p - p0)^2 + norm(q - q0)^2 + (r - r0)^2
err(x, y; atol = 1e-2, rtol = atol^2)  = isapprox(x, y; atol = atol, rtol = rtol) || x <= y + atol

function test_projection_simplex(p0; atol1::Real = 1e-2, atol2::Real = 1e-6)
	pe = Projections.simplex_exact(p0)
	p  = Projections.simplex(p0)

	@testset "simplex projection:" begin
		@test err(L(p,p0), L(pe,p0); atol = atol1)
		@test sum(p) ≈ 1  atol = atol2
		@test minimum(p) >= - atol2
		@test maximum(p) <= 1 + atol2
	end;
	return nothing
end

function test_projection_simplex_mod1(p0, q0, r0::Real, C1::Real, C2::Real; atol1::Real = 1e-2, atol2::Real = 1e-6)
	pe, qe, re = Projections.simplex_mod1_exact(p0, q0, r0, C1, C2)
	p, q, r    = Projections.simplex_mod1(p0, q0, r0, C1, C2)

	@testset "simplex mod1 projection:" begin
		@test err(L(p,p0,q,q0,r,r0), L(pe,p0,qe,q0,re,r0); atol = atol1)
		@test sum(p) ≈ sum(q)  atol = atol2
		@test minimum(p) >= - atol2
		@test minimum(q) >= - atol2
		@test maximum(p) <= C1 + atol2
		@test maximum(q) <= C2*r + atol2
	end;
	return nothing
end

function test_projection_simplex_mod2(p0, q0, C1::Real, C2::Integer; atol1::Real = 1e-2, atol2::Real = 1e-6)
	pe, qe = Projections.simplex_mod2_exact(p0, q0, C1, C2)
	p, q    = Projections.simplex_mod2(p0, q0, C1, C2)

	@testset "simplex mod2 projection:" begin
		@test err(L(p,p0,q,q0), L(pe,p0,qe,q0); atol = atol1)
		@test sum(p) ≈ sum(q)  atol = atol2
		@test minimum(p) >= - atol2
		@test minimum(q) >= - atol2
		@test maximum(p) <= C1 + atol2
		@test maximum(q) <= sum(p)/C2 + atol2
	end;
	return nothing
end

function test_minimize_linear_on_simplex(p0, c, ε::Real, k::Real; atol1::Real = 1e-2, atol2::Real = 1e-6)
	pe = Projections.minimize_linear_on_simplex_exact(p0, c, ε, k)
	if k == Inf
		p = Projections.minimize_linear_on_simplex_lInf(p0, c, ε)
	elseif k == 1
		p = Projections.minimize_linear_on_simplex_l1(p0, c, ε)
	elseif k == 2
		p = Projections.minimize_linear_on_simplex_l2(p0, c, ε)
	else
		@error "k ∉ {1, 2, Inf}"
		return nothing
	end

	@testset "minimize linear on simplex (l_$k norm):" begin
		@test c'*p ≈ c'*pe  atol = atol1

		@test c'*p ≈ c'*pe  atol = atol1
		@test sum(p) ≈ 1  atol = atol2
		@test minimum(p) >= - atol2
		@test norm(p - p0, k) <= ε + atol2
	end;
	return nothing
end


## Tests












# ## simplex mod 1
# Dp0 = [Uniform, Normal]
# Dq0 = [Uniform, Normal]
# r0s      = vcat(-100, -10, 0:0.1:1, 10, 100)
# ns       = [10, 100]
# ms       = [10, 100]
# C1s      = [0.001, 0.1, 1, 10, 100]
# C2s      = [0.001, 0.1, 1, 10, 100]
# # C1s = [0.001]
# # C2s = [0.01]
#
# allcombinations = Iterators.product(Dp0, Dq0, r0s, ns, ms, C1s, C2s)
#
# @testset "simplex mod 1" begin
# 	@testset "p0 = $dp0, q0 = $dq0, r0 = $r0, n = $n, m = $m, C1 = $C1, C2 = $C2" for (dp0, dq0, r0, n, m, C1, C2) in allcombinations
# 		p0  = rand(dp0(0,1), n)
# 		q0  = rand(dp0(0,1), m)
#
# 		C2 < m && test_projection_simplex_mod1(p0, q0, r0, C1, C2)
# 	end
# end;
#
# n  = 100
# m  = 10
# p0 = rand(Uniform(0,1), n)
# q0 = rand(Uniform(0,1), m)
# r0 = -100
# C1 = 0.001
# C2 = 0.001
#
# pe, qe,re = Projections.simplex_mod1_exact(p0, q0, r0, C1, C2)
# p, q,r   = Projections.simplex_mod1(p0, q0, r0, C1, C2)
#
# ## simplex mod 2
# Dp0 = [Uniform, Normal]
# Dq0 = [Uniform, Normal]
# ns       = [10, 100]
# ms       = [10, 100]
# C1s      = [0.001, 0.1, 1, 10, 100]
# C2s      = vcat(1:5, 10, 20, 99)
#
# allcombinations = Iterators.product(Dp0, Dq0, ns, ms, C1s, C2s)
#
# @testset "simplex mod 2" begin
# 	@testset "p0 = $dp0, q0 = $dq0, n = $n, m = $m, C1 = $C1, C2 = $C2" for (dp0, dq0, n, m, C1, C2) in allcombinations
# 		p0  = rand(dp0(0,1), n)
# 		q0  = rand(dq0(0,1), m)
#
# 		C2 < m && test_projection_simplex_mod2(p0, q0, C1, C2)
# 	end
# end;
#
#
# # n  = 100
# # m  = 10
# # p0 = rand(Uniform(0,1), n)
# # q0 = rand(Uniform(0,1), m)
# # C1 = 0.001
# # C2 = 0.01
# #
# # pe, qe = Projections.simplex_mod2_exact(p0, q0, C1, C2)
# # p, q   = Projections.simplex_mod2(p0, q0, C1, C2)
#
#
#
# function foo(k)
# 	n  = floor(Int64, 10^k/3)
# 	m  = floor(Int64, 2*10^k/3)
# 	p0 = rand(Uniform(0,1), n)
# 	q0 = rand(Uniform(0,1), m)
# 	C1 = 0.1
# 	C2 = floor(Int64, m/10)
# 	T = map(1:2) do i
# 		val, t, bytes, gctime, memallocs = @timed Projections.simplex_mod2(p0, q0, C1, C2)
# 		return [t  val[end]]
# 	end |> x -> vcat(x...)
#
# 	return mean(T, dims = 1)
# end
