function statstodataframe(stats::Stats)
    fields = fieldnames(Stats)
    vals   = [[getfield(stats, field)] for field in fields]
    return DataFrames.DataFrame(vals, [fields...])
end


function benchmark(d::Union{<:Divergence, <:Norm}, m::ModelDRO; maxevals::Int = 10, general::Bool = false)
    solve(d, m)

    rows = map(1:maxevals) do i
        general ? generalsolve(d, m) : solve(d, m)
        return statstodataframe(stats)
    end
    return DataFrames.aggregate(reduce(vcat, rows), [:model, :constraint, :optimizer], [Statistics.mean, Statistics.std])
end


function benchmark(d::Union{<:Divergence, <:Norm}, getmodel::Function, N::Vector{<:Int}; maxevals::Int = 10, general::Bool = false)
    solve(d, getmodel(10))

    rows = map(N) do n
        m = getmodel(n)
        return benchmark(d, m; maxevals = maxevals, general = general)
    end
    table           = reduce(vcat, rows)
    table.dimension = N

    return table
end

