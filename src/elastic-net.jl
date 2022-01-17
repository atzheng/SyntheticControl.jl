using JuMP
using Lasso
using Ipopt


function constrained_elastic_net(X::Array{T, 2}, y::Array{T, 1};
                                 λ=0., α=0.,
                                 solver=Ipopt.Optimizer,
                                 intercept=true,
                                 adding_up=false,
                                 non_negative=false,
                                 verbose=false,
                                 kwargs...) where T <: Real

    # Special cases with faster solve
    if !adding_up && !non_negative
        unconstrained_elastic_net(
            X, y; λ=λ, α=α, intercept=intercept, kwargs...)
    elseif α == 0.
        constrained_ridge_regression(
            X, y; λ=λ, solver=solver, intercept=intercept,
            adding_up=adding_up, non_negative=non_negative)
    else
        n, d = size(X)
        m = Model(solver)

        @variable(m, β0)
        @variable(m, βpos[1:d] ≥ 0)
        @variable(m, βneg[1:d] ≥ 0)

        # βabs and ε are not necessary but simplify statement
        # of the objective. This improves performance a lot.
        @variable(m, βabs[1:d] ≥ 0)
        @variable(m, ε[1:n])
        if verbose println("Starting to build objective...") end
        @objective(m, Min,
                   mean(ε .^ 2)
                   + λ * α * sum(βabs)
                   + λ * (1 - α) * sum(βabs .^ 2))
        @constraint(m, ε .== β0 .+ X * (βpos .- βneg) .- y)
        @constraint(m, βabs .== βpos .+ βneg)
        if !intercept @constraint(m, β0 == 0.) end
        if non_negative @constraint(m, βneg .== 0.) end
        if adding_up @constraint(m, sum(βpos .- βneg) == 1.) end
        if !verbose set_silent(m) end

        optimize!(m)
        value(β0), max.(value.(βpos), 0.) - max.(value.(βneg), 0.)
    end
end

function constrained_ridge_regression(X, y;
                                      λ=0.,
                                      solver=Ipopt.Optimizer,
                                      intercept=true,
                                      adding_up=false,
                                      non_negative=false,
                                      verbose=false)
    n, d = size(X)
    m = Model(solver)
    @variable(m, β0)
    @variable(m, β[1:d])
    @variable(m, ε[1:n])
    @objective(m, Min,
                     mean(ε .^ 2)
                     + λ * sum(β .^ 2))
    @constraint(m, ε .== β0 .+ X * β .- y)
    if !intercept @constraint(m, β0 == 0.) end
    if non_negative @constraint(m, β .>= 0.) end
    if adding_up @constraint(m, sum(β) == 1.) end
    if !verbose set_silent(m) end
    optimize!(m)
    βhat = non_negative ? max.(value.(β), 0.) : value.(β)
    value(β0), βhat
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
