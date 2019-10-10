function solve_exact(d::Divergence, m::Model; optimizer = Ipopt.Optimizer)

    model = JuMP.Model(JuMP.with_optimizer(optimizer, print_level = 0))

    ϕ = generate(d)
    JuMP.register(model, :ϕ, 1, ϕ, autodiff = true)

    JuMP.@variable(model, p[1:length(m.q)] >= 0)

    JuMP.@objective(model, Max, m.c'*p)
    JuMP.@constraint(model, sum(p) == 1)
    JuMP.@NLconstraint(model, sum(m.q[i]*ϕ(p[i]/m.q[i]) for i in eachindex(p)) <= m.ε)

    JuMP.optimize!(model)
    return JuMP.value.(p)
end


function solve(d::Divergence, m::Model)
  if check_ε(d, m)
    
    p = zero(m.q)
    p[m.Imax] .= m.q[m.Imax]
    p ./= sum(m.q[m.Imax])

    ϕ = generate(d)

    if sum(m.q .* ϕ.(p ./ m.q)) <= m.ε
      @info "plati veta"
      return p
    else
      h, bounds = find_mu(d,m)
      μ = Roots.bisection(h, bounds...)
      return optimal(d, m, μ)
    end
  else
    @info "wrong ε"
    return m.q
  end
end