function get_DRO(n::Int, d::Projections.Constraint, ε::Real = 0.1; seed::Int = 1234)
    Random.seed!(seed);    

    q = LinearAlgebra.normalize(rand(Distributions.Uniform(0,1), n), 1)
    c = rand(Distributions.Normal(0,1), n)
    return Projections.DRO(d, q, c, ε)
end


function get_Simplex1(n::Int; seed::Int = 1234)
    Random.seed!(seed);    

    q  = rand(n)
    lb = rand(n)./n
    ub = 1 .+ rand(n)
    return Projections.Simplex1(q, lb, ub)
end


function get_Simplex2(n::Int; seed::Int = 1234)
    Random.seed!(seed);    

    q  = rand(n)
    a  = rand(n)
    b  = 1 .+ rand(n)
    lb = rand(n)./n
    ub = 1 .+ rand(n)
    C1 = a'*(0.9*lb + 0.1*ub)
    C2 = b'*(0.9*lb + 0.1*ub)
    return Projections.Simplex2(q, lb, ub, a, b, C1, C2)
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
           m::T;
           title::String    = "h_$(typeof(m.d).name)",
           save::Bool       = false,
           savename::String = title,
           savepath::String = "",
           type::String     = "png") where {T <: Projections.Model}

    f(λ) = Projections.h(m.d, m, λ)

    p = plot(size = (900, 600), dpi = 150, legend = :best, show = false)
    title!(title)
    xlabel!("lambda")
    ylabel!("h(lambda)")
    plot!(λ, f.(λ))
    
    save && savefig(p, joinpath(savepath, "$(savename).$(type)"))
    return p
end


function maketable(table::DataFrame;
                   cols::Vector{<:Symbol} = [:constraint],
                   save::Bool       = false,
                   savepath::String = "",
                   savename::String = "comparison")
    
     maketable(table, :evaltime_mean; cols = cols, save = save, savepath = savepath, savename = savename)
     maketable(table, :evaltime_std;  cols = cols, save = save, savepath = savepath, savename = savename)
     maketable(table, :evals_mean;    cols = cols, save = save, savepath = savepath, savename = savename)
     maketable(table, :evals_std;     cols = cols, save = save, savepath = savepath, savename = savename)
    return 
end


function maketable(table::DataFrame,
                   metric::Symbol;
                   cols::Vector{<:Symbol} = [:constraint],
                   save::Bool       = false,
                   savepath::String = "",
                   savename::String = "comparison")

    table_new = DataFrames.DataFrame(dimension = sort(unique(table.dimension)))
    for df in DataFrames.groupby(table, cols)
        colname = join([string(df[1, col]) for col in cols], "_") 

        table_new[!, Symbol("$(colname)")] = df[!, metric]
    end

    save && CSV.write(joinpath(savepath,  "$(savename)_$(metric).csv"), table_new)
    return 
end


# ----------------------------------------------------------------------------------------------------------
# Divergences
# ----------------------------------------------------------------------------------------------------------
function comparison_divergences(N::AbstractVector{<:Int};
                                save::Bool = false,
                                savepath::String = "",
                                maxevals = 10)
    
    divergences = [Burg(), Hellinger(), ChiSquare(), ModifiedChiSquare(), KullbackLeibler()]

    function eval(d, N; verbose = true)
        return  Projections.benchmark(Sadda(), (n) -> get_DRO(n, d), N; verbose = verbose, maxevals = maxevals)
    end

    ## precompile for n
    map((d) -> eval(d, [10, 100]; verbose = false), divergences)
    
    ## evaluation
    table = mapreduce((d) -> eval(d, N), vcat, divergences)

    ## save csv
    maketable(table; save = save, savepath = savepath, savename = "comparison_divergences")

    ## create and save figures
    comparison(table, :evaltime; save = save, savepath = savepath, savename = "time_divergences")
    comparison(table, :evals;    save = save, savepath = savepath, savename = "evals_divergences")
    return table
end


# ----------------------------------------------------------------------------------------------------------
# Norms
# ----------------------------------------------------------------------------------------------------------
function comparison_norms(N::AbstractVector{<:Int};
                          save::Bool = false,
                          savepath::String = "",
                          maxevals = 10)
    
    norms = [Linf(), Lone(), Ltwo()]

    function eval(d, N; verbose = true)
        return  Projections.benchmark(Sadda(), (n) -> get_DRO(n, d), N; verbose = verbose, maxevals = maxevals)
    end

    ## precompile for n
    map((d) -> eval(d, [10, 100]; verbose = false), norms)
    
    ## evaluation
    table = mapreduce((d) -> eval(d, N), vcat, norms)

    ## save csv
    maketable(table; save = save, savepath = savepath, savename = "comparison_norms")

    ## create and save figures
    comparison(table, :evaltime; save = save, savepath = savepath, savename = "time_norms")
    comparison(table, :evals;    save = save, savepath = savepath, savename = "evals_norms")
    return table
end


# ----------------------------------------------------------------------------------------------------------
# Simplex
# ----------------------------------------------------------------------------------------------------------
function comparison_simplex(N::AbstractVector{<:Int};
                            save::Bool = false,
                            savepath::String = "",
                            maxevals = 10)

    function eval(N; verbose = true)
        row1 = Projections.benchmark(Sadda(), get_Simplex1, N; verbose = verbose, maxevals = maxevals)
        row2 = Projections.benchmark(Sadda(), get_Simplex2, N; verbose = verbose, maxevals = maxevals)
        return vcat(row1, row2)
    end

    ## precompile for n
    eval([10, 100]; verbose = false)
    
    ## evaluation
    table = eval(N)

    ## save csv
    cols = [:model]
    maketable(table; cols = cols, save = save, savepath = savepath, savename = "comparison_simplex")


    ## create and save figures
    comparison(table, :evaltime; save = save, savepath = savepath, savename = "time_simplex")
    comparison(table, :evals;    save = save, savepath = savepath, savename = "evals_simplex")
    return table
end



# ----------------------------------------------------------------------------------------------------------
# General solvers
# ----------------------------------------------------------------------------------------------------------
function comparison_general_solvers(N::AbstractVector{<:Int};
                                    save::Bool = false,
                                    savepath::String = "",
                                    maxevals = 10)

    constraints = [Linf(), Lone(), Ltwo(), 
                   KullbackLeibler(), Burg(), Hellinger(), ChiSquare(), ModifiedChiSquare()]
    
    function eval(d, N; save = false, verbose = true)
        t1    = Projections.benchmark(Sadda(),   (n) -> get_DRO(n, d), N; verbose = verbose, maxevals = maxevals)
        t2    = Projections.benchmark(General(), (n) -> get_DRO(n, d), N; verbose = verbose, maxevals = maxevals)
        table = vcat(t1, t2)

        comparison(table, :evaltime; save = save, savepath = savepath, savename = "time_$(typeof(d).name)_general_solvers")
        return  table
    end

    ## precompile for n
    map((d) -> eval(d, [10, 100]; save = false, verbose = false), constraints)
    
    ## evaluation
    table = mapreduce((d) -> eval(d, N; save = save), vcat, constraints)

    ## save csv
    cols = [:constraint, :solver]
    maketable(table; cols = cols, save = save, savepath = savepath, savename = "comparison_general_solvers")

    return table
end


# ----------------------------------------------------------------------------------------------------------
# Philpott
# ----------------------------------------------------------------------------------------------------------
function comparison_philpott(N::AbstractVector{<:Int};
                             save::Bool = false,
                             savepath::String = "",
                             maxevals = 10)

    solvers = [Projections.Sadda(), Projections.Philpott()]
    
    function eval(N; verbose = true)
        t1    = Projections.benchmark(Sadda(),    (n) -> get_DRO(n, Ltwo()), N; verbose = verbose, maxevals = maxevals)
        t2    = Projections.benchmark(Philpott(), (n) -> get_DRO(n, Ltwo()), N; verbose = verbose, maxevals = maxevals)
        return  vcat(t1, t2)
    end

    ## precompile for n
    eval([10, 100]; verbose = false)
    
    ## evaluation
    table = eval(N)

    ## save csv
    cols = [:constraint, :solver]
    maketable(table; cols = cols, save = save, savepath = savepath, savename = "comparison_philpott")

    ## create and save figures
    comparison(table, :evaltime; save = save, savepath = savepath, savename = "time_philpott")
    return table
end