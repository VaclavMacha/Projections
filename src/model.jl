"""
    DRO

A structure representing the DRO model.

Fields:
- `d::Constraint`:  type of the constraint
- `q::Vector`: probability distribution (vector)
- `c::Vector` vector of weights
- `ε::Real`: 
- `cmin::Real`: minimum of the vector c
- `cmax::Real`: maximum of the vector c
- `Imax::Vector`: set of indexes, which fulfill c[i] == cmax 
"""
struct DRO <: Model
    d::Constraint
    q::Vector
    c::Vector
    ε::Real
    cmin::Real
    cmax::Real
    Imax::Vector

    function DRO(d::Constraint, q::Vector{<:Real}, c::Vector{<:Real}, ε::Real)
        cmin = minimum(c)
        cmax = maximum(c)

        @assert length(q) == length(c)  "The length of the vector `q` and `c` must be the same."
        @assert all(q .>= 0)            "All components of the vector `q` must be nonnegative."
        @assert sum(q) ≈ 1 atol = 1e-10 "The `sum(q) = $(sum(q))` of the vector q must be equal to 1."
        @assert cmin != cmax            "The vector `c` must be non-constant."
        @assert ε > 0                   "The parameter `ε = $(ε)` must be greater than 0." 

        return new(d, q, c, ε, cmin, cmax, findall(c .== cmax))
    end
end


"""
    Simplex

A structure representing the DRO model.

Fields:

"""
struct Simplex <: Model
    q::Vector
    lb::Vector
    ub::Vector

    function Simplex(q::Vector{<:Real}, lb::Vector{<:Real}, ub::Vector{<:Real})

        @assert sum(lb) <= 1   "The feasible set is empty: sum(lb) = $(sum(lb)) > 1"
        @assert sum(ub) >= 1   "The feasible set is empty: sum(ub) = $(sum(ub)) < 1"
        @assert all(ub .>= lb) "Upper bounds `ub` mus be must greater or equl to lower bounds"

        return new(q, lb, ub)
    end
end