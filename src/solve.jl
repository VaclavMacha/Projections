# ----------------------------------------------------------------------------------------------------------
# our approach
# ----------------------------------------------------------------------------------------------------------
"""
    solve(d::Union{<:Divergence, <:Norm}, m::ModelDRO; kwargs...)

Solves model `m` with given ϕ-divergence or p-norm `d`. 
"""
function solve(d::Union{<:Divergence, <:Norm}, m::ModelDRO; kwargs...)
    reset_stats(d)
     p, t, bytes, gctime, memallocs = @timed optimal(d, m; kwargs...)
     add_stats(t, bytes)

    return p
end


# ----------------------------------------------------------------------------------------------------------
# general solvers
# ----------------------------------------------------------------------------------------------------------
"""
    generalsolve(d::Union{<:Divergence, <:Norm}, m::ModelDRO; kwargs...)

Solves model `m` with given ϕ-divergence or p-norm `d`. 
"""
function generalsolve(d::Union{<:Divergence, <:Norm}, m::ModelDRO)
    reset_stats(d)
    p, t, bytes, gctime, memallocs = @timed generaloptimal(d, m)
    add_stats(t, bytes)

    return p
end


"""
    generaloptimal(d::Divergence, m::ModelDRO)

Solves the given model `m` with ϕ-divergence `d` using the Ipopt solver. 
"""
function generaloptimal(d::Divergence, m::ModelDRO)
    stats.optimizer = "Ipopt"

    model = JuMP.Model(JuMP.with_optimizer(Ipopt.Optimizer, print_level = 0))

    ϕ = generate(d)
    JuMP.register(model, :ϕ, 1, ϕ, autodiff = true)

    JuMP.@variable(model, p[1:length(m.q)] >= 0)

    JuMP.@objective(model, Max, m.c'*p)
    JuMP.@constraint(model, sum(p) == 1)
    JuMP.@NLconstraint(model, sum(m.q[i]*ϕ(p[i]/m.q[i]) for i in eachindex(p)) <= m.ε)

    JuMP.optimize!(model)
    return JuMP.value.(p)
end


"""
    generaloptimal(d::Norm, m::ModelDRO)

Solves the given model `m` with norm `d` using the ECOS solver. 
"""
function generaloptimal(d::Norm, m::ModelDRO)
    stats.optimizer = "ECOS"

    k = normtype(d)
    p = Convex.Variable(length(m.q))

    objective   = m.c'*p
    constraints = [sum(p) == 1,
                   p >= 0,
                   Convex.norm(p - m.q, k) <= m.ε]

    problem = Convex.maximize(objective, constraints)
    Convex.solve!(problem, ECOS.ECOSSolver(verbose = false), verbose = false)
    return vec(p.value)
end