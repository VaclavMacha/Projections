using Projections, Test, Distributions, LinearAlgebra


include("/home/vaclav/GoogleDrive/projects/julia/2019/Projections/test/tests.jl")

# Test.@testset "All tests" begin
#     test_generate()
#     test_models()
# end


# test_models()

q = rand(10)
q ./= sum(q)
c = rand(10)
ε = 1e-6

d = Projections.Philpott()

m1 = Projections.Model(q,c,ε)
m2 = Projections.Model(sort(q; rev = true), sort(c), ε)

p1 = Projections.solve(d, m1);
p2 = Projections.solve_exact(d, m1);

# test_model(m1)
# test_model(m2) 

# d = Burg()
# ϕ = generate(d)
# h, bounds = find_mu(d,m)
# μ = Roots.bisection(h, bounds...)

# @time p1 = solve(d,m);
# @time p2 = solve_exact(d,m);

# BenchmarkTools.@btime p1 = solve(d,m);
# BenchmarkTools.@btime p2 = solve_exact(d,m);