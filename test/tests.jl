function divergences()
    return [Burg(),
            Hellinger(),
            ChiSquare(),
            ModifiedChiSquare(),
            KullbackLeibler()]
end


function norms()
    return [Linf(),
            Lone(),
            Ltwo(),
            Philpott()]
end


methods() = vcat(divergences(), norms())


function test_generate(; atol = 1e-4)
    Test.@testset "Test of generating functions" begin
        Test.@testset "$(Projections.name(d))" for d in divergences()
            ϕ = Projections.generate(d)
            Test.@test isapprox(ϕ(1), 0; atol = atol)
        end
    end
    return
end


function test_feasibility(d::Divergence, m::Model, p; atol = 1e-4)
    ϕ = Projections.generate(d)
    Test.@testset "Feasibility" begin
        Test.@test isapprox(sum(p), 1; atol = atol)
        Test.@test sum(m.q .* ϕ.(p ./ m.q)) <= 1.01 * m.ε
        Test.@test minimum(p) >= - atol
    end
    return
end


function test_feasibility(d::Norm, m::Model, p; atol = 1e-4)
    k = Projections.normtype(d)
    Test.@testset "Feasibility" begin
        Test.@test isapprox(sum(p), 1; atol = atol)
        Test.@test LinearAlgebra.norm(p - m.q, k) <= 1.01 * m.ε
        Test.@test minimum(p) >= - atol
    end
    return
end


function isfeasible(d::Divergence, m::Model, p; atol = 1e-4)
    ϕ = Projections.generate(d)
    return all([isapprox(sum(p), 1; atol = atol),
                sum(m.q.*ϕ.(p./m.q)) <= 1.01 * m.ε,
                minimum(p) >= - atol])
end


function isfeasible(d::Norm, m::Model, p; atol = 1e-4)
    k = Projections.normtype(d)
    return all([isapprox(sum(p), 1; atol = atol),
                LinearAlgebra.norm(p - m.q, k) <= 1.01 * m.ε,
                minimum(p) >= - atol])
end


function test_optimality(d::Union{Divergence, Norm}, m::Model, p, psolver; atol = 1e-4)
    isfeasible(d, m, psolver; atol = 1e-4) || return 
    
    Test.@testset "Optimality" begin
        Test.@test isapprox(m.c'*p, m.c'*psolver, atol = atol) || m.c'*p >= m.c'*psolver - atol
    end
    return
end


function test_model(m::Model; atol = 1e-4)
    Test.@testset "Comparison with the general solver" begin
        Test.@testset "$(Projections.name(d))" for d in methods()
            
            p       = solve(d,m);
            psolver = solve_exact(d,m);

            test_feasibility(d, m, p)
            test_optimality(d, m, p, psolver)
        end
    end
    return
end