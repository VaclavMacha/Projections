using Projections, Test, Distributions, LinearAlgebra

include("tests.jl")

@time @testset "All projections tests" begin
    ## simplex
    ns  = [10, 50, 100]
    Dp0 = [Uniform, Normal]

    allcombinations = Iterators.product(Dp0, ns)

    @testset "simplex" begin
        @testset "p0 = $dp0, n = $n" for (dp0, n) in allcombinations
            p0  = rand(dp0(0,1), n)
            test_projection_simplex(p0)
        end
    end;


    ## simplex mod 1
    Dp0 = [Uniform, Normal]
    Dq0 = [Uniform, Normal]
    r0s = vcat(-100, -10, 0:0.1:1, 10, 100)
    ns  = [10, 100]
    ms  = [10, 100]
    C1s = [0.001, 0.1, 1, 10, 100]
    C2s = [0.001, 0.1, 1, 10, 100]

    allcombinations = Iterators.product(Dp0, Dq0, r0s, ns, ms, C1s, C2s)

    @testset "simplex mod 1" begin
        @testset "p0 = $dp0, q0 = $dq0, r0 = $r0, n = $n, m = $m, C1 = $C1, C2 = $C2" for (dp0, dq0, r0, n, m, C1, C2) in allcombinations
            p0  = rand(dp0(0,1), n)
            q0  = rand(dp0(0,1), m)
            test_projection_simplex_mod1(p0, q0, r0, C1, C2)
        end
    end;


    ## simplex mod 2
    Dp0 = [Uniform, Normal]
    Dq0 = [Uniform, Normal]
    ns  = [10, 100]
    ms  = [10, 100]
    C1s = [0.001, 0.1, 1, 10, 100]
    C2s = vcat(1:5, 10, 20, 99)

    allcombinations = Iterators.product(Dp0, Dq0, ns, ms, C1s, C2s)

    @testset "simplex mod 2" begin
        @testset "p0 = $dp0, q0 = $dq0, n = $n, m = $m, C1 = $C1, C2 = $C2" for (dp0, dq0, n, m, C1, C2) in allcombinations
            p0  = rand(dp0(0,1), n)
            q0  = rand(dp0(0,1), m)

            C2 < m && test_projection_simplex_mod2(p0, q0, C1, C2)
        end
    end;


    ## simplex mod 3
    Dp0 = [Uniform, Normal]
    Dq0 = [Uniform, Normal]
    r0s = vcat(-100, -10, 0:0.1:1, 10, 100)
    ns  = [10, 100]
    ms  = [10, 100]

    allcombinations = Iterators.product(Dp0, Dq0, r0s, ns, ms)

    @testset "simplex mod 3" begin
        @testset "p0 = $dp0, q0 = $dq0, r0 = $r0, n = $n, m = $m" for (dp0, dq0, r0, n, m) in allcombinations
            p0  = rand(dp0(0,1), n)
            q0  = rand(dp0(0,1), m)
            test_projection_simplex_mod3(p0, q0, r0)
        end
    end;


    ## simplex mod 4
    Dp0 = [Uniform, Normal]
    Dq0 = [Uniform, Normal]
    ns  = [10, 100]
    ms  = [10, 100]
    Cs  = vcat(1:5, 10, 20, 99)

    allcombinations = Iterators.product(Dp0, Dq0, ns, ms, Cs)

    @testset "simplex mod 4" begin
        @testset "p0 = $dp0, q0 = $dq0, n = $n, m = $m, C = $C" for (dp0, dq0, n, m, C) in allcombinations
            p0  = rand(dp0(0,1), n)
            q0  = rand(dp0(0,1), m)

            C < m && test_projection_simplex_mod4(p0, q0, C)
        end
    end;


    ## minimize on linear simplex l1, l2 and lInf
    Dc  = [Uniform, Normal]
    Dp0 = [Uniform]
    εs  = vcat(1e-6:0.01:0.1, 0.2, 0.5, 1)
    n     = 100

    allcombinations1 = Iterators.product(Dc, Dp0, εs)
    allcombinations2 = Iterators.product(Dp0, εs)

    @testset "minimize on linear simplex: l1, l2, lInf norm" begin
        @testset "p0 = $dp0, c = $dc, ε = $ε" for (dc, dp0, ε) in allcombinations1
            c   = rand(dc(0,1), n)
            p0  = rand(dp0(0,1), n)
            p0 /= sum(p0)

            test_minimize_linear_on_simplex(p0, c, ε, Inf)
            test_minimize_linear_on_simplex(p0, c, ε, 1)
            test_minimize_linear_on_simplex(p0, c, ε, 2)
            test_philpott(p0, c, ε, :original)
            test_philpott(p0, c, ε, :optimized)

            sort!(c)
            sort!(p0; rev = true)

            @testset "sorted c, p0" begin
                test_minimize_linear_on_simplex(p0, c, ε, Inf)
                test_minimize_linear_on_simplex(p0, c, ε, 1)
                test_minimize_linear_on_simplex(p0, c, ε, 2)
                test_philpott(p0, c, ε, :original)
                test_philpott(p0, c, ε, :optimized)
            end
        end

        @testset "p0 = $dp0, c = max(min(Uniform, 0.8), 0.2), ε = $ε" for (dp0, ε) in allcombinations2
            c   = max.(min.(rand(Uniform(0,1), n), 0.8), 0.2)
            p0  = rand(dp0(0,1), n)
            p0 /= sum(p0)

            test_minimize_linear_on_simplex(p0, c, ε, Inf)
            test_minimize_linear_on_simplex(p0, c, ε, 1)
            test_minimize_linear_on_simplex(p0, c, ε, 2)
            test_philpott(p0, c, ε, :original)
            test_philpott(p0, c, ε, :optimized)
        end
    end;
end;
