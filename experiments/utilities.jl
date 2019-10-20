function get_DRO(n::Int, d::Projections.Constraint, ε::Real = 0.1; seed::Int = 1234)
    Random.seed!(seed);    

    q = LinearAlgebra.normalize(rand(Distributions.Uniform(0,1), n), 1)
    c = rand(Distributions.Normal(0,1), n)
    return Projections.DRO(d, q, c, ε)
end


function comparison(table::DataFrames.DataFrame,
                    varname::Symbol  = :evaltime;
                    xscale::Symbol   = :identity,
                    yscale::Symbol   = :identity,
                    ploterror::Bool  = false,
                    title::String    = "$(varname) comparison",
                    save::Bool       = false,
                    savename::String = title,
                    savepath::String = "",
                    type::String     = "png")

    cols = [:model, :constraint, :solver, :optimizer]

    p = plot(size = (900, 600), dpi = 150, legend = :topleft)
    title!(title)
    xlabel!("n")
    ylabel!(string(varname))
    for df in DataFrames.groupby(table, cols)
        label = join([df[1, col] for col in cols], " - ")

        n     = df.dimension
        t     = df[!, Symbol(varname, :_mean)]
        t_std = df[!, Symbol(varname, :_std)]

        if ploterror
            plot!(n, t, yerror = t_std, label = label, xscale = xscale, yscale = yscale)
        else
            plot!(n, t, label = label, xscale = xscale, yscale = yscale)
        end
    end

    save && savefig(p, joinpath(savepath, "$(savename).$(type)"))
    return p
end



function h(λ::AbstractVector{<:Real},
           m::Projections.Model;
           title::String    = "h_$(typeof(m.d).name)",
           save::Bool       = false,
           savename::String = title,
           savepath::String = "",
           type::String     = "png")

    f(λ) = Projections.h(m.d,m,λ)

    p = plot(size = (900, 600), dpi = 150, legend = :best, show = true)
    title!(title)
    xlabel!("lambda")
    ylabel!("h(lambda)")
    plot!(λ, f.(λ))
    
    save && savefig(p, joinpath(savepath, "$(savename).$(type)"))
    return p
end


# ----------------------------------------------------------------------------------------------------------
# Divergences
# ----------------------------------------------------------------------------------------------------------
function comparison_divergences(N::AbstractVector{<:Int} = 25000:25000:1000000;
                                save::Bool = false,
                                savepath::String = "")
    
    divergences = [Burg(), Hellinger(), ChiSquare(), ModifiedChiSquare(), KullbackLeibler()]

    function eval(d, N; verbose = true)
        return  Projections.benchmark(Sadda(), (n) -> get_DRO(n, d), N; verbose = verbose)
    end

    ## precompile for n
    map((d) -> eval(d, [10, 100]; verbose = false), divergences)
    
    ## evaluation
    table = mapreduce((d) -> eval(d, N), vcat, divergences)

    ## save csv
    save && CSV.write(joinpath(savepath,  "comparison_divergences.csv"), table)

    ## create and save figures
    comparison(table, :evaltime; save = save, savepath = savepath, savename = "time_divergences")
    comparison(table, :evals; save = save, savepath = savepath, savename = "evals_divergences")
    return table
end


# ----------------------------------------------------------------------------------------------------------
# Norms
# ----------------------------------------------------------------------------------------------------------
function comparison_norms(N::AbstractVector{<:Int} = 25000:25000:1000000;
                          save::Bool = false,
                          savepath::String = "")
    
    norms = [Linf(), Lone(), Ltwo()]

    function eval(d, N; verbose = true)
        return  Projections.benchmark(Sadda(), (n) -> get_DRO(n, d), N; verbose = verbose)
    end

    ## precompile for n
    map((d) -> eval(d, [10, 100]; verbose = false), norms)
    
    ## evaluation
    table = mapreduce((d) -> eval(d, N), vcat, norms)

    ## save csv
    save && CSV.write(joinpath(savepath,  "comparison_norms.csv"), table)

    ## create and save figures
    comparison(table, :evaltime; save = save, savepath = savepath, savename = "time_norms")
    comparison(table, :evals; save = save, savepath = savepath, savename = "evals_norms")
    return table
end


# ----------------------------------------------------------------------------------------------------------
# General solvers
# ----------------------------------------------------------------------------------------------------------
function comparison_general_solvers(N::AbstractVector{<:Int} = 250:250:10000;
                                    save::Bool = false,
                                    savepath::String = "")

    constraints = [Linf(), Lone(), Ltwo(), 
                   KullbackLeibler(), Burg(), Hellinger(), ChiSquare(), ModifiedChiSquare()]
    
    function eval(d, N; save = false, verbose = true)
        t1    = Projections.benchmark(Sadda(),   (n) -> get_DRO(n, d), N; verbose = verbose)
        t2    = Projections.benchmark(General(), (n) -> get_DRO(n, d), N; verbose = verbose)
        table = vcat(t1, t2)

        comparison(table, :evaltime; save = save, savepath = savepath, savename = "time_$(typeof(d).name)_general_solvers")
        return  table
    end

    ## precompile for n
    map((d) -> eval(d, [10, 100]; save = false, verbose = false), constraints)
    
    ## evaluation
    table = mapreduce((d) -> eval(d, N; save = save), vcat, constraints)

    ## save csv
    save && CSV.write(joinpath(savepath,  "comparison_general_solvers.csv"), table)
    return table
end


# ----------------------------------------------------------------------------------------------------------
# Philpott
# ----------------------------------------------------------------------------------------------------------
function comparison_philpott(N::AbstractVector{<:Int} = 250:250:10000;
                             save::Bool = false,
                             savepath::String = "")

    solvers = [Projections.Sadda(), Projections.Philpott()]
    
    function eval(N; verbose = true)
        t1    = Projections.benchmark(Sadda(),    (n) -> get_DRO(n, Ltwo()), N; verbose = verbose)
        t2    = Projections.benchmark(Philpott(), (n) -> get_DRO(n, Ltwo()), N; verbose = verbose)
        return  vcat(t1, t2)
    end

    ## precompile for n
    eval([10, 100]; verbose = false)
    
    ## evaluation
    table = eval(N)

    ## save csv
    save && CSV.write(joinpath(savepath,  "comparison_philpott.csv"), table)

    ## create and save figures
    comparison(table, :evaltime; save = save, savepath = savepath, savename = "time_philpott")
    return table
end