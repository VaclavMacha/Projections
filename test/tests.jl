
function divergences()
    return [Projections.Burg(),
            Projections.Hellinger(),
            Projections.ChiSquare(),
            Projections.ModifiedChiSquare(),
            Projections.KullbackLeibler()]
end


function test_generate(; atol = 1e-4)
    Test.@testset "Test of generating functions" begin
        Test.@testset "$(Projections.name(d))" for d in divergences()
            ϕ = Projections.generate(d)
            Test.@test isapprox(ϕ(1), 0; atol = atol)
        end
    end
    return
end


function test_model(m::Projections.Model; atol = 1e-4)
    Test.@testset "Comparison with the general solver" begin
        Test.@testset "$(Projections.name(d))" for d in divergences()
            ϕ  = Projections.generate(d)
            p1 = Projections.solve(d,m);
            p2 = Projections.solve_exact(d,m);

            Test.@testset "Feasibility" begin
                Test.@test isapprox(sum(p1), 1; atol = atol)
                Test.@test sum(m.q.*ϕ.(p1./m.q)) <= 1.01 * m.ε
                Test.@test minimum(p1) >= - atol
            end
            cond = [isapprox(sum(p2), 1; atol = atol),
                    sum(m.q.*ϕ.(p2./m.q)) <= 1.01 * m.ε,
                    minimum(p2) >= - atol]
            if all(cond)
                Test.@testset "Optimality" begin
                    Test.@test m.c'*p1 >= m.c'*p2 - atol
                end
            end
        end
    end
    return
end


function test_models(; n::Integer = 100)
    εs = vcat(1e-6:0.01:0.1, 0.2, 0.5, 1)

    Test.@testset "DRO" begin
        Test.@testset "ε = $ε" for ε in εs
            q   = rand(n)
            q ./= sum(q)
            c1  = rand(n)
            c2  = max.(min.(rand(n), 0.8), 0.2)

            Test.@testset "standard" begin
                m = Projections.Model(q, c1, ε)
                test_model(m)
            end
            Test.@testset "sorted" begin
                m = Projections.Model(sort(q; rev = true), sort(c1), ε)
                test_model(m) 
            end
            Test.@testset "bounds" begin
                m = Projections.Model(q, c2, ε)
                test_model(m)
            end
        end 
    end
    return 
end