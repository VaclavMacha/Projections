## Pat@Mat ##
## projection - hinge loss
function g_recursion(β0::AbstractArray{<:Real}, ϑ2::Real)
	n      = length(β0)
	g      = zero(β0)
	g[end] = β0[end]

	for j = n-1:-1:1
		g[j] = g[j+1] - (ϑ2^2*(n - j) +1)*(β0[j+1] - β0[j])
	end
	return g
end


function δhat(λ::Real, g::AbstractArray{<:Real}, β0::AbstractArray{<:Real}, δ0::Real, ϑ2::Real)
	n  = length(β0)
	ub = searchsortedfirst(g, ϑ2*δ0 - λ)

	if ub == 1
	 	δhat = (ϑ2*δ0 - λ + ϑ2^2*sum(β0))/(ϑ2^2*n + 1)
	elseif ub == n + 1
	 	δhat = ϑ2*δ0 - λ
	else
	 	lb = ub - 1
	 	δhat = (β0[ub] - β0[lb])/(g[ub] - g[lb])*(ϑ2*δ0 - λ - g[lb]) + β0[lb]
	end
	return δhat
end


function PatMat_projection_hinge(α0::AbstractArray{<:Real}, β0::AbstractArray{<:Real}, δ0::Real, C::Real, ϑ1::Real, ϑ2::Real)

	if δ0 <= - sum(max.(β0 .+ maximum(α0), 0))
		α = zero(α0)
		β = zero(β0)
		δ = zero(δ0)
	else
		perm = sortperm(β0)
		β0   = β0[perm]
		g    = g_recursion(β0, ϑ2)
		δ    = 0

		function f(λ)
			δ = (δhat(λ, g, β0, δ0, ϑ2) + λ)/ϑ2
			return sum(min.(max.(α0 .- λ, 0), ϑ1*C)) - sum(min.(max.(β0 .+ λ, 0), ϑ2*δ))
		end

		λ = 0
		try
			λ  = Roots.find_zero(f, (maximum(α0) - maximum(β0))/2, Order1())
		catch
			@warn "Secant method failed -> Bisection"
			λ  = Roots.find_zero(f, (-maximum(β0), maximum(α0)), Bisection())
		end
		β0 = β0[invperm(perm)]

		α  = @. min(max(α0 - λ, 0), ϑ1*C)
		β  = @. min(max(β0 + λ, 0), ϑ2*δ)
	end
	return α, β, δ
end


## projection - truncated quadratic loss
function PatMat_projection_quadratic(α0::AbstractArray{<:Real}, β0::AbstractArray{<:Real}, δ0::Real)

	if -maximum(β0) > maximum(α0)
		α = zero(α0)
		β = zero(β0)
		δ = zero(δ0)
	else
		f(λ) = sum(max.(α0 .- λ, 0)) - sum(max.(β0 .+ λ, 0))

		λ = 0
		try
			λ  = Roots.find_zero(f, (maximum(α0) - maximum(β0))/2, Order1())
		catch
			@warn "Secant method failed -> Bisection"
			λ  = Roots.find_zero(f, (-maximum(β0), maximum(α0)), Bisection())
		end
		α = @. max(α0 - λ, 0)
		β = @. max(β0 + λ, 0)
		δ = max(δ0, 0.001)
	end
	return α, β, δ
end


##=============================================================================================================
## TopRank ##
## projection - hinge loss
function find_λ(α_bar, k, γs)
	i, j, d  = 2, 1, 1
	γ, γ_old = γs[1], 0
	g, g_old = 0, 0

	@inbounds while g < k*α_bar
		g_old = g
		γ_old = γ
		γj_α  = γs[j] + α_bar
		γi    = γs[i]

		if γi <= γj_α
			g += d*(γi - γ)
			γ = γi
			d += 1
			i += 1
		else
			g += d*(γj_α - γ)
			γ = γj_α
			d -= 1
			j += 1
		end
	end
	λ = (γ - γ_old)/(g - g_old)*(k*α_bar - g_old) + γ_old
	return λ
end


function  TopRank_projection_hinge(α0::AbstractArray{<:Real}, β0::AbstractArray{<:Real}, k::Integer, C::Real, ϑ::Real)

	if mean(partialsort(β0, 1:k; rev = true)) + maximum(α0) <= 0
		α = zero(α0)
		β = zero(β0)
	else
		γs = vcat(.- sort(β0; rev = true), Inf)

		function fα_bar(α_bar)
			λ = find_λ(α_bar, k, γs)
			return sum(min.(max.(α0 .- λ .+ sum(max.(β0 .+ λ .- α_bar, 0))/k, 0), ϑ*C)) - k*α_bar
		end

		λ     = 0
		α_bar = 0
		n     = length(β0)
		try
			α_bar = Roots.find_zero(fα_bar, min(10, n*ϑ*C/(2*k)), Order1())
			α_bar <= 0 && @error("Wrong solution")
		catch
			@warn "Secant method failed -> Bisection"
			α_bar = Roots.find_zero(fα_bar, (1e-10, n*ϑ*C/k), Bisection())
		end

		ρ = sum(max.(β0 .+ λ .- α_bar, 0))/k
		α = @. max(min(α0 - λ + ρ, ϑ*C), 0)
		β = @. max(min(β0 + λ, α_bar), 0)
	end
	return α, β
end


## projection - truncated quadratic
function TopRank_projection_quadratic(α0::AbstractArray{<:Real}, β0::AbstractArray{<:Real}, k::Integer)

	if mean(partialsort(β0, 1:k; rev = true)) + maximum(α0) <= 0
		α = zero(α0)
		β = zero(β0)
	else
		γs = vcat(.- sort(β0; rev = true), Inf)

		function fα_bar(α_bar)
			λ = find_λ(α_bar, k, γs)
			return sum(max.(α0 .- λ .+ sum(max.(β0 .+ λ .- α_bar, 0))/k, 0)) - k*α_bar
		end
		λ     = 0
		α_bar = 0
		try
			α_bar = Roots.find_zero(fα_bar, 10, Order1())
			α_bar <= 0 && @error("Wrong solution")
		catch
			n     = length(β0)
			α_bar = Roots.find_zero(fα_bar, (1e-10, (sum(α0) + sum(β0))/k), Bisection())
		end


		ρ = sum(max.(β0 .+ λ .- α_bar, 0))/k
		α = @. max(α0 - λ + ρ, 0)
		β = @. max(min(β0 + λ, α_bar), 0)
	end
	return α, β, α_bar, λ
end
