function statstodataframe(stats::Stats)
    fields = fieldnames(Stats)
    vals   = [[getfield(stats, field)] for field in fields]
    return DataFrames.DataFrame(vals, [fields...])
end


function benchmark(s::Solver, m::Model; maxevals::Int = 10)
    solve(s, m)

    table = map(1:maxevals) do i
        solve(s, m)
        return statstodataframe(stats)
    end |> rows -> reduce(vcat, rows)
    return DataFrames.aggregate(table, [:model, :constraint, :solver, :optimizer], [Statistics.mean, Statistics.std])
end


function benchmark(s::Solver, getmodel::Function, N::Vector{<:Int}; maxevals::Int = 10)
    solve(s, getmodel(10))

    table = map(N) do n
        m = getmodel(n)
        return benchmark(s, m; maxevals = maxevals)
    end |> rows -> reduce(vcat, rows)
    table.dimension = N

    return table
end

