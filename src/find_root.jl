function find_root(f, x0, xlb, xub; method::Symbol = :try, atol::Real = 0, verbose::Bool = false)
    reset_stats!()

    if method == :secant
        return find_root_secant(f, x0, xlb, xub; atol = atol, verbose = verbose)
    elseif method == :bisection
        return find_root_bisection(f, x0, xlb, xub; atol = atol, verbose = verbose)
    elseif method == :try
        try
            x = find_root_secant(f, x0, xlb, xub; atol = atol, verbose = verbose)
            if isnan(x)
                error("Secant method failed")
            else
                return x
            end
        catch
            verbose && @warn "Secant method failed -> Bisection method will be used."
            return find_root_bisection(f, x0, xlb, xub; atol = atol, verbose = verbose)
        end
    else
        @error "$method method not defined. Use method ∈ {:try, :secant, :bisection}."
    end
end


function find_root_secant(f, x0, xlb, xub; atol::Real = 1e-8, verbose::Bool = false)
    new_key!(:secant)
    x = Roots.secant_method(f, x0; atol = atol)

    if isnan(x)
        verbose && @warn "Secant method failed"
        return NaN
    elseif x < xlb || x > xub
        verbose && @warn "Secant method returned infeasible solution."
        return NaN
    else
        return x
    end
end


function find_root_bisection(f, x0, xlb, xub; atol::Real = 1e-8, verbose::Bool = false)
    new_key!(:bisection)
    x = Roots.bisection(f, xlb, xub; xatol = atol)

    if isnan(x)
        verbose && @warn "Bisection method failed"
        return NaN
    else
        return x
    end
end

# function find_root_falseposition(f, x0, xlb, xub; atol::Real = 1e-8, verbose::Bool = false)
#     update_key!(:falseposition)
#     x = Roots.find_zero(f, (xlb, xub), FalsePosition(); atol = atol)
#
#     if isnan(x)
#         verbose && @warn "FalsePosition method failed"
#         error("FalsePosition method failed")
#         return NaN
#     else
#         return x
#     end
# end
