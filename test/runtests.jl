using Projections, Test, Distributions, LinearAlgebra

include("tests.jl")

n  = 100
εs = vcat(1e-6:0.01:0.1, 0.2, 0.5, 1)

Test.@testset "All tests" begin
    test_generate()

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
end

