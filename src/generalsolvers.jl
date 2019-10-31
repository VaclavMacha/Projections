"""
    optimal(s::General, d::KullbackLeibler, m::DRO)

Solves the DRO model `m` with Kullback-Leibler divergence using the Ipopt solver. 
"""
function optimal(s::General, d::KullbackLeibler, m::DRO)
    stats.optimizer = "Ipopt"

    model = JuMP.Model(JuMP.with_optimizer(Ipopt.Optimizer, print_level = 0))

    JuMP.@variable(model, p[1:length(m.q)] >= 0)

    JuMP.@objective(model, Max, m.c'*p)

    JuMP.@constraint(model, sum(p) == 1)
    JuMP.@NLconstraint(model, sum(p[i]*log(p[i]/m.q[i]) for i in eachindex(p)) <= m.ε)

    JuMP.optimize!(model)
    return JuMP.value.(p)
end


"""
    optimal(s::General, d::Burg, m::DRO)

Solves the DRO model `m` with Burg entropy using the Ipopt solver. 
"""
function optimal(s::General, d::Burg, m::DRO)
    stats.optimizer = "Ipopt"

    model = JuMP.Model(JuMP.with_optimizer(Ipopt.Optimizer, print_level = 0))

    JuMP.@variable(model, p[1:length(m.q)] >= 0)

    JuMP.@objective(model, Max, m.c'*p)

    JuMP.@constraint(model, sum(p) == 1)
    JuMP.@NLconstraint(model, sum(m.q[i]*log(m.q[i]/p[i]) for i in eachindex(p)) <= m.ε)

    JuMP.optimize!(model)
    return JuMP.value.(p)
end


"""
    optimal(s::General, d::Hellinger, m::DRO)

Solves the DRO model `m` with Hellinger distance using the Ipopt solver. 
"""
function optimal(s::General, d::Hellinger, m::DRO)
    stats.optimizer = "Ipopt"

    model = JuMP.Model(JuMP.with_optimizer(Ipopt.Optimizer, print_level = 0))

    JuMP.@variable(model, p[1:length(m.q)] >= 0)

    JuMP.@objective(model, Max, m.c'*p)

    JuMP.@constraint(model, sum(p) == 1)
    JuMP.@NLconstraint(model, sum((sqrt(p[i]) - sqrt(m.q[i]))^2 for i in eachindex(p)) <= m.ε)

    JuMP.optimize!(model)
    return JuMP.value.(p)
end


"""
    optimal(s::General, d::ChiSquare, m::DRO)

Solves the DRO model `m` with χ²-distance using the Ipopt solver. 
"""
function optimal(s::General, d::ChiSquare, m::DRO)
    stats.optimizer = "Ipopt"

    model = JuMP.Model(JuMP.with_optimizer(Ipopt.Optimizer, print_level = 0))

    JuMP.@variable(model, p[1:length(m.q)] >= 0)

    JuMP.@objective(model, Max, m.c'*p)

    JuMP.@constraint(model, sum(p) == 1)
    JuMP.@NLconstraint(model, sum(((p[i] - m.q[i])^2)/p[i] for i in eachindex(p)) <= m.ε)

    JuMP.optimize!(model)
    return JuMP.value.(p)
end


"""
    optimal(s::General, d::ModifiedChiSquare, m::DRO)

Solves the DRO model `m` with modified χ²-distance using the Ipopt solver. 
"""
function optimal(s::General, d::ModifiedChiSquare, m::DRO)
    stats.optimizer = "Ipopt"

    model = JuMP.Model(JuMP.with_optimizer(Ipopt.Optimizer, print_level = 0))

    JuMP.@variable(model, p[1:length(m.q)] >= 0)

    JuMP.@objective(model, Max, m.c'*p)

    JuMP.@constraint(model, sum(p) == 1)
    JuMP.@NLconstraint(model, sum(((p[i] - m.q[i])^2)/m.q[i] for i in eachindex(p)) <= m.ε)

    JuMP.optimize!(model)
    return JuMP.value.(p)
end


"""
    optimal(s::General, d::Linf, m::DRO)
 
Solves the DRO model `m` with l-infinity using the CPLEX solver. 
"""
function optimal(s::General, d::Linf, m::DRO)
    stats.optimizer = "CPLEX"

    model = JuMP.Model(JuMP.with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND=0))

    JuMP.@variable(model, p[1:length(m.q)] >= 0)
    JuMP.@variable(model, p_max >= 0)

    JuMP.@objective(model, Max, m.c'*p)
    
    JuMP.@constraint(model, p_max .>= p - m.q)
    JuMP.@constraint(model, p_max .>= - p + m.q)
    JuMP.@constraint(model, sum(p) == 1)
    JuMP.@constraint(model, p_max <= m.ε)

    JuMP.optimize!(model)
    return JuMP.value.(p)
end


"""
    optimal(s::General, d::Lone, m::DRO)

Solves the DRO model `m` with l-1 using the CPLEX solver. 
"""
function optimal(s::General, d::Lone, m::DRO)
    stats.optimizer = "CPLEX"

    model = JuMP.Model(JuMP.with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND=0))

    JuMP.@variable(model, p[1:length(m.q)] >= 0)
    JuMP.@variable(model, p_abs[1:length(m.q)] >= 0)
    
    JuMP.@objective(model, Max, m.c'*p)
    
    JuMP.@constraint(model, p_abs .>= p - m.q)
    JuMP.@constraint(model, p_abs .>= - p + m.q)
    JuMP.@constraint(model, sum(p) == 1)
    JuMP.@constraint(model, sum(p_abs) <= m.ε)

    JuMP.optimize!(model)
    return JuMP.value.(p)
end


"""
    optimal(s::General, d::Ltwo, m::DRO)

Solves the DRO model `m` with l-2 using the CPLEX solver. 
"""
function optimal(s::General, d::Ltwo, m::DRO)
    stats.optimizer = "CPLEX"

    model = JuMP.Model(JuMP.with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND=0))

    JuMP.@variable(model, p[1:length(m.q)] >= 0)
    
    JuMP.@objective(model, Max, m.c'*p)
    JuMP.@constraint(model, sum(p) == 1)
    JuMP.@constraint(model, sum((p[i] - m.q[i])^2 for i in eachindex(p)) <= m.ε^2)

    JuMP.optimize!(model)
    return JuMP.value.(p)
end


"""
    optimal(s::General, m::Simplex)

Solves the Simplex model using the CPLEX solver. 
"""
function optimal(s::General, m::Simplex)
    stats.optimizer = "CPLEX"

    model = JuMP.Model(JuMP.with_optimizer(CPLEX.Optimizer, CPX_PARAM_SCRIND=0))

    JuMP.@variable(model, m.lb[i] <= p[i = 1:length(m.q)] <= m.ub[i])
    
    JuMP.@objective(model, Min, sum(0.5 * (p[i] - m.q[i])^2 for i in eachindex(p)))
    JuMP.@constraint(model, sum(p) == 1)

    JuMP.optimize!(model)
    return JuMP.value.(p)
end