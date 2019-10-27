# ----------------------------------------------------------------------------------------------------------
# DRO models
# ----------------------------------------------------------------------------------------------------------
function test_generate(d::Projections.Divergence; atol::Real = 1e-4)
    Test.@testset "$(Projections.name(d))" begin
        ϕ = Projections.generate(d)
        Test.@test isapprox(ϕ(1), 0; atol = atol)
    end
    return
end


function test_feasibility(m::Projections.DRO, p; atol::Real = 1e-6)
    Test.@testset "Feasibility" begin
        Test.@test isapprox(sum(p), 1; atol = atol)
        Test.@test minimum(p) >= - atol
        if typeof(m.d) <: Projections.Divergence
            ϕ     = Projections.generate(m.d)
            Test.@test sum(m.q .* ϕ.(p ./ m.q)) <= 1.01 * m.ε
        else
            k     = Projections.normtype(m.d)
            Test.@test LinearAlgebra.norm(p - m.q, k) <= 1.01 * m.ε
        end
    end
    return
end


function isfeasible(m::Projections.DRO, p; atol::Real = 1e-6)

    cond1 = isapprox(sum(p), 1; atol = atol)
    cond2 = minimum(p) >= - atol
    if typeof(m.d) <: Projections.Divergence
        ϕ     = Projections.generate(m.d)
        cond3 = sum(m.q.*ϕ.(p./m.q)) <= 1.01 * m.ε
    else
        k     = Projections.normtype(m.d)
        cond3 = LinearAlgebra.norm(p - m.q, k) <= 1.01 * m.ε
    end
    return all([cond1, cond2, cond3])
end


function test_optimality(m::Projections.DRO, p, psolver; atol = 1e-4)
    isfeasible(m, psolver; atol = atol) || return 
    
    Test.@testset "Optimality" begin
        Test.@test m.c'*p >= m.c'*psolver - atol
    end
    return
end


function test_model(m::Projections.DRO)
    Test.@testset "Comparison with the general solver" begin 
        p       = Projections.solve(Projections.Sadda(), m);
        psolver = Projections.solve(Projections.General(), m);

        test_feasibility(m, p; atol = 1e-6)
        test_optimality(m, p, psolver; atol = 1e-4)

        if typeof(m.d) <: Projections.Ltwo
            p = Projections.solve(Projections.Philpott(), m);

            test_feasibility(m, p; atol = 1e-6)
            test_optimality(m, p, psolver; atol = 1e-4)
        end
    end
    return
end


# ----------------------------------------------------------------------------------------------------------
# Simplex models
# ----------------------------------------------------------------------------------------------------------
function test_feasibility(m::T, p; atol::Real = 1e-6) where {T<:Projections.Simplex}
    Test.@testset "Feasibility" begin
        Test.@test all(p .>= m.lb .- atol)
        Test.@test all(p .<= m.ub .+ atol)
        if T <: Projections.Simplex1
            Test.@test isapprox(sum(p), 1; atol = atol)
        else
            Test.@test isapprox(m.a'*p, m.C1; atol = atol)
            Test.@test isapprox(m.b'*p, m.C2; atol = atol)
        end
    end
    return
end


function isfeasible(m::T, p; atol::Real = 1e-6) where {T<:Projections.Simplex}

    cond1 = all(p .>= m.lb .- atol)
    cond2 = all(p .<= m.ub .+ atol)
    if T <: Projections.Simplex1
        cond3 = isapprox(sum(p), 1; atol = atol)
        cond4 = true
    else
        cond3 = isapprox(m.a'*p, m.C1; atol = atol)
        cond4 = isapprox(m.b'*p, m.C2; atol = atol)
    end
    return all([cond1, cond2, cond3, cond4])
end


function test_optimality(m::Projections.Simplex, p, psolver; atol = 1e-4)
    isfeasible(m, psolver; atol = atol) || return 
    
    Test.@testset "Optimality" begin
        Test.@test LinearAlgebra.norm(p .- m.q)^2 <= LinearAlgebra.norm(psolver .- m.q)^2 + atol
    end
    return
end


function test_model(m::Projections.Simplex)
    Test.@testset "Comparison with the general solver" begin 
        p       = Projections.solve(Projections.Sadda(), m);
        psolver = Projections.solve(Projections.General(), m);

        test_feasibility(m, p; atol = 1e-6)
        test_optimality(m, p, psolver; atol = 1e-4)
    end
    return
end