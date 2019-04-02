## Projection tests
function test_projection_feasibility(method::String,
									 surrogate::String,
									 α::Array{<:Real},
									 β::Array{<:Real};
									 δ::Real    = NaN,
									 k::Integer = 0,
									 α_lb::Real = -Inf,
									 α_ub::Real = Inf,
									 β_lb::Real = -Inf,
									 β_ub::Real = Inf,
									 atol::Real = 1e-3,
									 kwargs...)

	@testset "$method projection for $surrogate loss - feasibility: " begin
		@testset "∑α = ∑β: " begin
			@test ≈(sum(α), sum(β), atol = atol)
		end
		@testset "α[$i]: " for i in 1:length(α)
			@test α_lb <= α[i] <= α_ub
		end
		@testset "β[$i]: " for i in 1:length(β)
			@test β_lb <= β[i] <= β_ub
		end
		if !isnan(δ)
			@testset "δ: " begin
				@test 0 <= δ
			end
		end
	end
end


function test_projection_optimality(method::String,
									surrogate::String,
									α::Array{<:Real},
									α_opt::Array{<:Real},
									β::Array{<:Real},
									β_opt::Array{<:Real},
									δ::Real     = NaN,
									δ_opt::Real = NaN;
									atol::Real = 1e-3,
									kwargs...)

	labels    = vec(vcat(["α[$i]" for i in 1:length(α)], ["β[$i]" for i in 1:length(β)]))
	val_opt   = vec(vcat(α_opt, β_opt))
	val 	  = vec(vcat(α, β))
	if !isnan(δ)
		push!(labels, "δ")
		push!(val_opt, δ_opt)
		push!(val, δ)
	end
	rel_error = testerror.(val, val_opt)

	try
		@testset "$method projection for $surrogate loss - optimality: " begin
			@testset "relative error α[$i]: " for i in 1:length(α)
				@test ≈(rel_error[i], 0, atol = atol)
			end
			@testset "relative error β[$i]: " for i in length(α) .+ 1:length(β)
				@test ≈(rel_error[i], 0, atol = atol)
			end
			if !isnan(δ)
				@testset "relative error δ: " begin
					@test ≈(rel_error[end-1], 0, atol = atol)
				end
			end
		end;
	catch
		nothing
	end
	return DataFrame(hcat(labels, val_opt, val, rel_error), [:labels, :val_opt, :val, :relative_error])
end


##=============================================================================================================
## Pat@Mat projection - hinge loss
function test_PatMat_projection_hinge(α0::Vector{<:Real} = 8*rand(100) .- 1,
							          β0::Vector{<:Real} = 6*rand(150) .- 1,
							          δ0::Real = 2*rand() - 1,
							          C::Real = 2;
							          ϑ1::Real = 2,
							          ϑ2::Real = 0.1,
							          kwargs...)

	α_opt, β_opt, δ_opt = Projections.PatMat_projection_exact_hinge(α0, β0, δ0, C, ϑ1, ϑ2)
	α, β, δ 			= Projections.PatMat_projection_hinge(α0, β0, δ0, C, ϑ1, ϑ2)

	test_projection_feasibility("PatMat", "hinge", α, β; δ = δ, α_lb = 0, α_ub = ϑ1*C, β_lb = 0, β_ub = ϑ2*δ, kwargs...)
	return test_projection_optimality("PatMat", "hinge", α, α_opt, β,  β_opt, δ, δ_opt; kwargs...)
end


## Pat@Mat projection - truncated quadratic loss
function test_PatMat_projection_quadratic(α0::Vector{<:Real} = 8*rand(100) .- 1,
								          β0::Vector{<:Real} = 6*rand(150) .- 1,
								          δ0::Real = 2*rand() - 1;
								          kwargs...)

	α_opt, β_opt, δ_opt = Projections.PatMat_projection_exact_quadratic(α0, β0, δ0)
	α, β, δ 			= Projections.PatMat_projection_quadratic(α0, β0, δ0)

	test_projection_feasibility("PatMat", "truncated quadratic", α, β; δ = δ, α_lb = 0, β_lb = 0, kwargs...)
	return test_projection_optimality("PatMat", "truncated quadratic", α, α_opt, β,  β_opt, δ, δ_opt; kwargs...)
end


##=============================================================================================================
## TopRank projection - hinge loss
function test_TopRank_projection_hinge(α0::Vector{<:Real} = 8*rand(100) .- 1,
							           β0::Vector{<:Real} = 6*rand(150) .- 1,
							           k::Integer = 5,
							           C::Real = 2;
							           ϑ::Real = 0.1,
							           kwargs...)

	α_opt, β_opt = Projections.TopRank_projection_exact_hinge(α0, β0, k, C, ϑ)
	α, β 		 = Projections.TopRank_projection_hinge(α0, β0, k, C, ϑ)

	test_projection_feasibility("TopRank", "hinge", α, β; k = k, α_lb = 0, α_ub = ϑ*C, β_lb = 0, β_ub = sum(α)/k, kwargs...)
	return test_projection_optimality("TopRank", "hinge", α, α_opt, β, β_opt; kwargs...)
end


## Pat@Mat projection - truncated quadratic loss
function test_TopRank_projection_quadratic(α0::Vector{<:Real} = 8*rand(100) .- 1,
								           β0::Vector{<:Real} = 6*rand(150) .- 1,
								           k::Integer = 5;
								           kwargs...)

	α_opt, β_opt = Projections.TopRank_projection_exact_quadratic(α0, β0, k)
	α, β 		 = Projections.TopRank_projection_quadratic(α0, β0, k)

	test_projection_feasibility("TopRank", "quadratic", α, β; k = k, α_lb = 0, β_lb = 0, β_ub = sum(α)/k, kwargs...)
	return test_projection_optimality("TopRank", "quadratic", α, α_opt, β, β_opt; kwargs...)
end
