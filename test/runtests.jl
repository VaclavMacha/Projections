import Projections, Test, Distributions, LinearAlgebra

include("tests.jl")

divergences = [Projections.Burg(), Projections.Hellinger(), Projections.ChiSquare(),
               Projections.ModifiedChiSquare(), Projections.KullbackLeibler()]
norms       = [Projections.Linf(), Projections.Lone(), Projections.Ltwo()]
constraints = vcat(divergences, norms)        

n  = 100
εs = vcat(1e-6:0.01:0.1, 0.2, 0.5, 1)

Test.@testset "All tests" begin

    Test.@testset "Test of generating functions" for d in divergences
        test_generate(d)
    end 

    Test.@testset "DRO" begin
        Test.@testset "ε = $ε" for ε in εs
            Test.@testset "Constraint = $(Projections.name(d))" for d in constraints
                q   = LinearAlgebra.normalize(rand(n), 1)
                c1  = rand(n)
                c2  = max.(min.(rand(n), 0.8), 0.2)

                Test.@testset "standard" begin
                    m = Projections.DRO(d, q, c1, ε)
                    test_model(m)
                end
                Test.@testset "sorted" begin
                    m = Projections.DRO(d, sort(q; rev = true), sort(c1), ε)
                    test_model(m) 
                end
                Test.@testset "bounds" begin
                    m = Projections.DRO(d, q, c2, ε)
                    test_model(m)
                end
            end
        end 
    end
end

