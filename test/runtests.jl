using Projections, Test, Distributions, LinearAlgebra

include("tests.jl")

Test.@testset "All tests" begin
    test_generate()
    test_models()
end