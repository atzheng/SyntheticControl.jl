using JuMP
using Lasso
using Ipopt


function constrained_elastic_net(X::Array{T, 2}, y::Array{T, 1};
                                 λ=0., α=0.,
                                 solver=nothing,
                                 intercept=true,
                                 adding_up=false,
                                 non_negative=false,
                                 verbose=false,
                                 kwargs...) where T <: Real

    if solver === nothing
        if !adding_up && !non_negative
            # Use faster solver
            return unconstrained_elastic_net(
                X, y; λ=λ, α=α, intercept=intercept, kwargs...)
        else
            solver = Ipopt.Optimizer
        end
    end

    d = size(X, 2)
    m = Model(solver)

    @variable(m, β0)
    @variable(m, βpos[1:d] ≥ 0)
    @variable(m, βneg[1:d] ≥ 0)
    @objective(m, Min,
               mean((β0 .+ X * (βpos .- βneg) .- y) .^ 2)
               + λ * α * sum(βpos .+ βneg)
               + λ * (1 - α) * sum((βpos .+ βneg) .^ 2))

    if !intercept @constraint(m, β0 == 0.) end
    if non_negative @constraint(m, βneg .== 0.) end
    if adding_up @constraint(m, sum(βpos .- βneg) == 1.) end
    if !verbose set_silent(m) end

    optimize!(m)
    value(β0), max.(value.(βpos), 0.) - max.(value.(βneg), 0.)
end

function unconstrained_elastic_net(X::Array{T, 2}, y::Array{T, 1};
                                   λ=0., α=0., intercept=true,
                                   kwargs...) where T <: Real
    βhat = coef(fit(LassoPath, X, y;
                    standardize=false, intercept=intercept, α=α, λ=[λ],
                    kwargs...))
    βhat[1], βhat[2:end]
end

function elastic_net_sc(O::Array{S, 2}, Z::Array{S, 2};
                        Tpre=nothing, kwargs...) where S <: Real
    N, T = size(O)
    is_control, Tpre_inferred = infer_controls(Z)
    Tpre = Tpre !== nothing ? Tpre : Tpre_inferred

    Oco_pre = O[is_control, 1:Tpre]
    # FIXME this is not really the right way to deal with multiple treatment units
    Otr_pre = mean(O[.!is_control, 1:Tpre]; dims=1) |> vec
    ω0, ω = constrained_elastic_net(collect(Oco_pre'), Otr_pre; kwargs...)
    Ohat_post = ω'O[is_control, Tpre+1:end] .+ ω0
    Otr_post = O[.!is_control, Tpre+1:end]
    wmean(Otr_post .- Ohat_post, Z[.!is_control, Tpre+1:end])
end
