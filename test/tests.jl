L(p, p0, q = 0, q0 = 0, r = 0, r0 = 0) = norm(p - p0)^2 + norm(q - q0)^2 + (r - r0)^2
err(x, y; atol = 1e-2, rtol = atol^2)  = isapprox(x, y; atol = atol, rtol = rtol) || x <= y + atol


function test_simplex(p0; atol1::Real = 1e-2, atol2::Real = 1e-6)
    pe = Projections.solve_simplex_exact(p0)
    p  = Projections.solve_simplex(p0)

    @testset "simplex projection:" begin
        @test err(L(p,p0), L(pe,p0); atol = atol1)
        @test sum(p) ≈ 1  atol = atol1
        @test minimum(p) >= - atol2
        @test maximum(p) <= 1 + atol2
    end;
end


function test_AATP1(p0, q0, r0::Real, C1::Real, C2::Real; atol1::Real = 1e-2, atol2::Real = 1e-6)
    pe, qe, re = Projections.solve_AATP1_exact(p0, q0, r0, C1, C2)
    p, q, r    = Projections.solve_AATP1(p0, q0, r0, C1, C2)

    @testset "AATP1 projection:" begin
        @test err(L(p,p0,q,q0,r,r0), L(pe,p0,qe,q0,re,r0); atol = atol1)
        @test sum(p) ≈ sum(q)  atol = atol1
        @test minimum(p) >= - atol2
        @test minimum(q) >= - atol2
        @test maximum(p) <= C1 + atol2
        @test maximum(q) <= C2*r + atol2
    end;
end


function test_AATP2(p0, q0, C1::Real, C2::Integer; atol1::Real = 1e-2, atol2::Real = 1e-6)
    pe, qe = Projections.solve_AATP2_exact(p0, q0, C1, C2)
    p, q   = Projections.solve_AATP2(p0, q0, C1, C2)

    @testset "AATP2 projection:" begin
        @test err(L(p,p0,q,q0), L(pe,p0,qe,q0); atol = atol1)
        @test sum(p) ≈ sum(q)  atol = atol1
        @test minimum(p) >= - atol2
        @test minimum(q) >= - atol2
        @test maximum(p) <= C1 + atol2
        @test maximum(q) <= sum(p)/C2 + atol2
    end;
end


function test_AATP3(p0, q0, r0::Real; atol1::Real = 1e-2, atol2::Real = 1e-6)
    pe, qe, re = Projections.solve_AATP3_exact(p0, q0, r0)
    p, q, r    = Projections.solve_AATP3(p0, q0, r0)

    @testset "AATP3 projection:" begin
        @test err(L(p,p0,q,q0,r,r0), L(pe,p0,qe,q0,re,r0); atol = atol1)
        @test sum(p) ≈ sum(q)  atol = atol1
        @test minimum(p) >= - atol2
        @test minimum(q) >= - atol2
    end;
end


function test_AATP4(p0, q0, C::Integer; atol1::Real = 1e-2, atol2::Real = 1e-6)
    pe, qe = Projections.solve_AATP4_exact(p0, q0, C)
    p, q   = Projections.solve_AATP4(p0, q0, C)

    @testset "AATP4 projection:" begin
        @test err(L(p,p0,q,q0), L(pe,p0,qe,q0); atol = atol1)
        @test sum(p) ≈ sum(q)  atol = atol1
        @test minimum(p) >= - atol2
        @test minimum(q) >= - atol2
        @test maximum(q) <= sum(p)/C + atol2
    end;
end


function test_DRO(p0, c, ε::Real, k::Real; atol1::Real = 1e-2, atol2::Real = 1e-6)
    pe = Projections.solve_DRO_exact(p0, c, ε, k)
    if k == Inf
        p = Projections.solve_DRO_lInf(p0, c, ε)
    elseif k == 1
        p = Projections.solve_DRO_l1(p0, c, ε)
    elseif k == 2
        p = Projections.solve_DRO_l2(p0, c, ε)
    else
        @error "k ∉ {1, 2, Inf}"
        return nothing
    end

    @testset "DRO (l_$k norm):" begin
        @test c'*p ≈ c'*pe  atol = atol1

        @test c'*p ≈ c'*pe  atol = atol1
        @test sum(p) ≈ 1  atol = atol1
        @test minimum(p) >= - atol2
        @test norm(p - p0, k) <= ε + atol2
    end;
end

function test_philpott(p0, c, ε::Real, type::Symbol; atol1::Real = 1e-2, atol2::Real = 1e-6)
    pe = Projections.solve_DRO_exact(p0, c, ε, 2)
    if type == :original
        p = Projections.solve_philpott(p0, c, ε)
    elseif type == :optimized
        p = Projections.solve_philpott_optimized(p0, c, ε)
    else
        @error "type ∉ {:original, :optimized}"
        return nothing
    end

    @testset "philpot $type:" begin
        @test c'*p ≈ c'*pe  atol = atol1
        @test sum(p) ≈ 1  atol = atol1
        @test minimum(p) >= - atol2
        @test norm(p - p0, 2) <= ε + atol2
    end;
end
