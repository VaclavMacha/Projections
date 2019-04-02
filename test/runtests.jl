using Projections, Test, DataFrames

include("test_projections.jl")

function testerror(x, x0)
	abs_error = abs(x - x0)
	max_abs   = max(abs(x), abs(x0))
	rel_error = iszero(max_abs) ? 0 : abs_error/max_abs
	return min(abs_error, rel_error)
end

@time @testset "Pat@Mat:" begin
	@time test_PatMat_projection_hinge()
	@time test_PatMat_projection_quadratic()
end;

@time @testset "TopRank:" begin
	@time test_TopRank_projection_hinge()
	@time test_TopRank_projection_quadratic()
end;
