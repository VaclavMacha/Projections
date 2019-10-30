# ----------------------------------------------------------------------------------------------------------
# Model generation
# ----------------------------------------------------------------------------------------------------------
function model_DRO(n::Int, d::Projections.Constraint, ε::Real = 0.1; seed::Int = 1234)
    Random.seed!(seed);    

    q = LinearAlgebra.normalize(rand(Distributions.Uniform(0,1), n), 1)
    c = rand(Distributions.Normal(0,1), n)
    return Projections.DRO(d, q, c, ε)
end


function model_Simplex1(n::Int; seed::Int = 1234)
    Random.seed!(seed);    

    q  = rand(n)
    lb = rand(n)./n
    ub = 1 .+ rand(n)
    return Projections.Simplex1(q, lb, ub)
end


function model_Simplex2(n::Int; seed::Int = 1234)
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


# ----------------------------------------------------------------------------------------------------------
# Benchmark evaluation
# ----------------------------------------------------------------------------------------------------------
function eval_Divergences(solver::Solver, N; kwargs...)
    divergences = [KullbackLeibler(), Hellinger(), Burg(), ChiSquare(), ModifiedChiSquare()]

    return  mapreduce((d) -> eval_DRO(solver, d, N; kwargs...), vcat, divergences)
end


function eval_Norms(solver::Solver, N; kwargs...)
    norms = [Linf(), Lone(), Ltwo()]

    return  mapreduce((d) -> eval_DRO(solver, d, N; kwargs...), vcat, norms)
end


function eval_DRO(solver::Solver, d, N; kwargs...)
    return  Projections.benchmark(solver, (n) -> model_DRO(n, d), N; kwargs...)
end


function eval_Simplex1(solver::Solver, N; kwargs...)
    return Projections.benchmark(solver, model_Simplex1, N; kwargs...)
end


function eval_Simplex2(solver::Solver, N; kwargs...)
    return Projections.benchmark(solver, model_Simplex2, N; kwargs...) 
end


# ----------------------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------------------
function tableformat(metric::Symbol)
    colnames = [:dimension, :dimension_small]
    for problem in [:KullbackLeibler, :Hellinger, :Burg, :ChiSquare, :ModifiedChiSquare]
        push!(colnames, Symbol(problem, "_Sadda"))
        metric in [:evals_mean, :evals_std] || push!(colnames, Symbol(problem, "_General"))
    end

    if metric in [:evals_mean, :evals_std] 
        push!(colnames, :Ltwo_Sadda)
        push!(colnames, :Simplex2_Sadda)
    else
        for problem in [:Linf, :Lone, :Ltwo, :Simplex1, :Simplex2]
            push!(colnames, Symbol(problem, "_Sadda"))
            push!(colnames, Symbol(problem, "_General"))
            problem == :Ltwo && push!(colnames, Symbol(problem, "_Philpott"))
        end
    end
    return colnames
end

function savetable(table::DataFrame;
                   save::Bool       = false,
                   savepath::String = "",
                   savename::String = "comparison")
    
    savetable(table, :evaltime_mean; save = save, savepath = savepath, savename = savename)
    savetable(table, :evaltime_std;  save = save, savepath = savepath, savename = savename)
    savetable(table, :evals_mean;    save = save, savepath = savepath, savename = savename)
    savetable(table, :evals_std;     save = save, savepath = savepath, savename = savename)
    
    save && CSV.write(joinpath(savepath,  "$(savename)_full_dataframe.csv"), table)

    return 
end


function savetable(table::DataFrame,
                   metric::Symbol;
                   save::Bool       = false,
                   savepath::String = "",
                   savename::String = "comparison")

    cols      = [:model, :constraint, :solver]

    dimension = sort(unique(table.dimension))    
    n         = length(dimension)
    table_new = DataFrames.DataFrame(dimension = dimension)

    dimension_small = sort(unique(table.dimension[table.solver .== "General"]))
    l               = length(dimension_small)
    table_new[!, :dimension_small] = vcat(dimension_small, [missing for i in 1:n-l])

    for df in DataFrames.groupby(table, cols)
        colname  = join([string(df[1, col]) for col in cols], "_") 
        colvalue = df[!, metric]
        l        = length(colvalue)

        table_new[!, Symbol("$(colname)")] = vcat(colvalue, [missing for i in 1:n-l])
    end

    colnames = [replace(colname, "DRO_" => "")  for colname in string.(names(table_new))]
    colnames = [replace(colname, "none_" => "") for colname in colnames]
    names!(table_new, Symbol.(colnames))

    table_new = table_new[!, tableformat(metric)]

    save && CSV.write(joinpath(savepath,  "$(savename)_$(metric).csv"), table_new)
    return table_new
end


# ----------------------------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------
# Comparison solver
# ----------------------------------------------------------------------------------------------------------
function comparison_solver(solver::T,
                           N::AbstractVector{<:Int};
                           save::Bool        = false,
                           savepath::String  = "",
                           maxevals::Integer = 10) where {T<:Solver}
    
    ## DRO with divergences
    @info "DRO with divergences"
    table1 = eval_Divergences(solver, N; verbose = true,  maxevals = maxevals)

    ## DRO with norms
    @info "DRO with norms"
    table2 = eval_Norms(solver, N; verbose = true,  maxevals = maxevals)

    ## Simplex1
    @info "Simplex1 model"
    table3 = eval_Simplex1(solver, N; verbose = true,  maxevals = maxevals)

    ## Simplex2
    @info "Simplex2 model"
    table4 = eval_Simplex2(solver, N; verbose = true,  maxevals = maxevals)

    ## Plots - time comparison
    comparison(table2, :evaltime; save = save, savepath = savepath, savename = "time_norms_$(T.name)")
    comparison(table1, :evaltime; save = save, savepath = savepath, savename = "time_divergences_$(T.name)")
    comparison(vcat(table3, table4), :evaltime; save = save, savepath = savepath, savename = "time_simplex_$(T.name)")

    ## Plots - #objective function evaluation
    if T <: Sadda
        comparison(table2, :evals;    save = save, savepath = savepath, savename = "evals_norms_$(T.name)")
        comparison(table1, :evals;    save = save, savepath = savepath, savename = "evals_divergences_$(T.name)")
        comparison(vcat(table3, table4), :evals;    save = save, savepath = savepath, savename = "evals_simplex_$(T.name)")
    end

    return vcat(table1, table2, table3, table4)
end


# ----------------------------------------------------------------------------------------------------------
# Comparison Philpott
# ----------------------------------------------------------------------------------------------------------
function comparison_philpott(N::AbstractVector{<:Int};
                             save::Bool        = false,
                             savepath::String  = "",
                             maxevals::Integer = 10)
    
    ## DRO with l2 norm and Philpott() solver
    @info "DRO with l2 norm and Philpott() solver"
    table = eval_DRO(Philpott(), Ltwo(), N; verbose = true,  maxevals = maxevals)

    ## Plots - time comparison
    comparison(table, :evaltime; save = save, savepath = savepath, savename = "time_philpott")
    return table
end