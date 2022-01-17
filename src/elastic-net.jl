using JuMP
using Ipopt

DEFAULT_SOLVER = Ipopt.Optimizer


function constrained_elastic_net(X::Array{T, 2}, y::Array{T, 1};
                                 λ1=0., λ2=0.,
                                 solver=DEFAULT_SOLVER,
                                 intercept=true,
                                 adding_up=false,
                                 non_negative=false) where T <: Real
    d = size(X, 2)
    m = Model(solver)

    @variable(m, β0)
    @variable(m, βpos[1:d] ≥ 0)
    @variable(m, βneg[1:d] ≥ 0)
    @objective(m, Min,
               mean((β0 .+ X * (βpos .- βneg) .- y) .^ 2)
               + λ1 * sum(βpos .+ βneg)
               + λ2 * sum((βpos .+ βneg) .^ 2))

    if !intercept @constraint(m, β0 == 0.) end
    if non_negative @constraint(m, βneg .== 0.) end
    if adding_up @constraint(m, sum(βpos .- βneg) == 1.) end

    optimize!(m)
    value(β0), max.(value.(βpos), 0.) - max.(value.(βneg), 0.)
end


function elastic_net_sc(O::Array{S, 2}, Z::Array{S, 2};
                        Tpre=nothing, kwargs...) where S <: Real
    N, T = size(O)
    is_control, Tpre_inferred = infer_controls(Z)
    Tpre = Tpre !== nothing ? Tpre : Tpre_inferred

    constrained_elastic_net
    Oco_pre = O[is_control, 1:Tpre]
    # FIXME this is not really the right way to deal with multiple treatment units
    Otr_pre = mean(O[.!is_control, 1:Tpre]; dims=1) |> vec
    ω0, ω = constrained_elastic_net(collect(Oco_pre'),
                                    Otr_pre; kwargs...)
    Ohat_post = ω'O[is_control, Tpre+1:end] .+ ω0
    Otr_post = O[.!is_control, Tpre+1:end]
    wmean(Otr_post .- Ohat_post, Z[.!is_control, Tpre+1:end])
end
