"""
    ModelDRO

A structure representing the DRO model.

Fields:
- `q::Vector{Real}`: probability distribution (vector)
- `c::Vector{Real}` vector of weights
- `ε::Real`: 
- `cmin::Real`: minimum of the vector c
- `cmax::Real`: maximum of the vector c
- `Imax::Vector{Integer}`: set of indexes, which fulfill c[i] == cmax 
"""
struct ModelDRO
    q::Vector
    c::Vector
    ε::Real
    cmin::Real
    cmax::Real
    Imax::Vector
end

"""
    ModelDRO(q::Vector{<:Real}, c::Vector{<:Real}, ε::Real)

A constructor of the ModelDRO structure.
"""
function ModelDRO(q::Vector{<:Real}, c::Vector{<:Real}, ε::Real)
    cmin = minimum(c)
    cmax = maximum(c)

    @assert length(q) == length(c) "The length of the vector `q` and `c` must be the same."
    @assert all(q .>= 0) "All components of the vector `q` must be nonnegative."
    @assert isapprox(sum(q), 1; atol = 1e-10)  "The sum of the vector q must be equal to 1."
    @assert cmin != cmax "The vector `c` must be non-constant."
    @assert ε > 0 "The parameter `ε` must be greater than 0." 

    return ModelDRO(q, c, ε, cmin, cmax, findall(c .== cmax))
end