"""
    optimal(s::General, d::Divergence, m::DRO)

Solves the given model `m` with ϕ-divergence `d` using the Ipopt solver. 
"""
function optimal(s::General, d::Divergence, m::DRO)
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
    optimal(s::General, d::Linf, m::DRO)

Solves the given model `m` with l-infinity norm `d` using the CPLEX solver. 
"""
function optimal(s::General, d::Linf, m::DRO)
    stats.optimizer = "CPLEX"

    model = JuMP.Model(JuMP.with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND=0))

    JuMP.@variable(model, p[1:length(m.q)] >= 0)
    JuMP.@variable(model, p_max >= 0)
    JuMP.@constraint(model, p_max .>= p - m.q)
    JuMP.@constraint(model, p_max .>= - p + m.q)
    JuMP.@constraint(model, sum(p) == 1)
    JuMP.@constraint(model, p_max <= m.ε)
    JuMP.@objective(model, Max, m.c'*p)

    JuMP.optimize!(model)
    return JuMP.value.(p)
end


"""
    optimal(s::General, d::Lone, m::DRO)

Solves the given model `m` with l-1 norm `d` using the CPLEX solver. 
"""
function optimal(s::General, d::Lone, m::DRO)
    stats.optimizer = "CPLEX"

    model = JuMP.Model(JuMP.with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND=0))

    JuMP.@variable(model, p[1:length(m.q)] >= 0)
    JuMP.@variable(model, p_abs[1:length(m.q)] >= 0)
    JuMP.@constraint(model, p_abs .>= p - m.q)
    JuMP.@constraint(model, p_abs .>= - p + m.q)
    JuMP.@constraint(model, sum(p) == 1)
    JuMP.@constraint(model, sum(p_abs) <= m.ε)
    JuMP.@objective(model, Max, m.c'*p)

    JuMP.optimize!(model)
    return JuMP.value.(p)
end


"""
    optimal(s::General, d::Ltwo, m::DRO)

Solves the given model `m` with l-2 norm `d` using the ECOS solver. 
"""
function optimal(s::General, d::Ltwo, m::DRO)
    stats.optimizer = "ECOS"

    p = Convex.Variable(length(m.q))

    objective   = m.c'*p
    constraints = [sum(p) == 1,
                   p >= 0,
                   Convex.norm(p - m.q, 2) <= m.ε]

    problem = Convex.maximize(objective, constraints)
    Convex.solve!(problem, ECOS.ECOSSolver(verbose = false), verbose = false)
    return vec(p.value)
end