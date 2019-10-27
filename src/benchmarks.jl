function getinfo(s::Solver, m::DRO)
    return "Benchmark of the $(typeof(s).name) solver for the $(typeof(m).name) model with $(name(m.d)) \n"
end

function getinfo(s::Solver, m::Simplex)
    return "Benchmark of the $(typeof(s).name) solver for the $(typeof(m).name) model \n"
end



function statstodataframe(stats::Stats)
    fields = fieldnames(Stats)
    vals   = [[getfield(stats, field)] for field in fields]
    return DataFrames.DataFrame(vals, [fields...])
end


function benchmark(s::Solver, m::Model; maxevals::Int = 10, verbose::Bool = true)
    verbose && printstyled(getinfo(s, m); bold = true, color = :green)
    solve(s, m)

    progress = ProgressMeter.Progress(maxevals, 0.1)

    table = map(1:maxevals) do i
        solve(s, m)
        row = statstodataframe(stats)
        verbose && ProgressMeter.next!(progress)
        return row
    end |> rows -> reduce(vcat, rows)

    return DataFrames.aggregate(table, [:model, :constraint, :solver, :optimizer], [Statistics.mean, Statistics.std])
end


function benchmark(s::Solver, getmodel::Function, N::AbstractVector{<:Int}; maxevals::Int = 10, verbose::Bool = true)
    m = getmodel(10)
    verbose && printstyled(getinfo(s, m); bold = true, color = :green)
    solve(s, m)

    progress = ProgressMeter.Progress(length(N), 0.1)

    table = map(N) do n
        m   = getmodel(n)
        row = benchmark(s, m; maxevals = maxevals, verbose = false)
        verbose && ProgressMeter.next!(progress)
        return row
    end |> rows -> reduce(vcat, rows)
    table.dimension = N

    return table
end