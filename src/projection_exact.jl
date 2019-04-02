## Pat@Mat ##
## exact projection - hinge loss 
function PatMat_projection_exact_hinge(α0::AbstractArray{<:Real}, β0::AbstractArray{<:Real}, δ0::Real, C::Real, ϑ1::Real, ϑ2::Real)
	α = Convex.Variable(length(α0))
	β = Convex.Variable(length(β0))
	δ = Convex.Variable()

	objective   = sumsquares(α - α0)/2 + sumsquares(β - β0)/2 + sumsquares(δ - δ0)/2
	constraints = [sum(α) == sum(β),
				   α <= ϑ1*C,
				   β <= ϑ2*δ,
				   α >= 0,
				   β >= 0] 
	
	problem = Convex.minimize(objective, constraints)
	Convex.solve!(problem, SCSSolver(verbose = 0))

	return α.value, β.value, δ.value
end


## exact projection - truncated quadratic loss ##
function PatMat_projection_exact_quadratic(α0::AbstractArray{<:Real}, β0::AbstractArray{<:Real}, δ0::Real)
	α = Convex.Variable(length(α0))
	β = Convex.Variable(length(β0))
	δ = Convex.Variable()

	objective   = sumsquares(α - α0)/2 + sumsquares(β - β0)/2 + sumsquares(δ - δ0)/2
	constraints = [sum(α) == sum(β),
				   α >= 0,
				   β >= 0,
				   δ >= 0] 
	
	problem = Convex.minimize(objective, constraints)
	Convex.solve!(problem, SCSSolver(verbose = 0))

	return α.value, β.value, δ.value
end


##=============================================================================================================
## TopRank ##
## exact projection - hinge loss 
function TopRank_projection_exact_hinge(α0::AbstractArray{<:Real}, β0::AbstractArray{<:Real}, k::Integer,  C::Real, ϑ::Real)
	α = Convex.Variable(length(α0))
	β = Convex.Variable(length(β0))

	objective   = sumsquares(α - α0)/2 + sumsquares(β - β0)/2
	constraints = [sum(α) == sum(β),
				   β <= sum(α)/k,
				   α <= ϑ*C,
				   α >= 0,
				   β >= 0] 
	
	problem = Convex.minimize(objective, constraints)
	Convex.solve!(problem, SCSSolver(verbose = 0))

	return α.value, β.value
end


## exact projection - truncated quadratic loss
function TopRank_projection_exact_quadratic(α0::AbstractArray{<:Real}, β0::AbstractArray{<:Real}, k::Integer)
	α = Convex.Variable(length(α0))
	β = Convex.Variable(length(β0))

	objective   = sumsquares(α - α0)/2 + sumsquares(β - β0)/2
	constraints = [sum(α) == sum(β),
				   β <= sum(α)/k,
				   α >= 0,
				   β >= 0] 
	
	problem = Convex.minimize(objective, constraints)
	Convex.solve!(problem, SCSSolver(verbose = 0))

	return α.value, β.value
end