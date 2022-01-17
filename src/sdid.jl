using StatsBase

function SDID(O::Array{S, 2}, Z::Array{S, 2};
              Tpre=nothing, solver=DEFAULT_SOLVER, kwargs...) where S <: Real
    """
    Implements Synthetic Differences-In-Differences (SDID) from
    Arkhangelsky et. al., 2021. Notation generally follows theirs.
    """
    # Get sizes
    N, T = size(O)
    is_control, Tpre_inferred = infer_controls(Z)
    Tpre = Tpre !== nothing ? Tpre : Tpre_inferred
    Nco = sum(is_control)
    Ntr = N - Nco
    Tpost = T - Tpre

    # 1. Compute regularizer ζ
    Δ = O[is_control, 2:Tpre] - O[is_control, 1:Tpre-1]
    σ2 = var(Δ[:, 1:Tpre-1]; corrected=false)
    ζ2 = sqrt(Ntr * Tpost) * σ2

    # 2. Solve for unit weights
    Opre = O[is_control, 1:Tpre]
    ω0, ωco = constrained_elastic_net(
        collect(Opre'),
        vec(mean(O[.!is_control, 1:Tpre]; dims=1));
        λ2=ζ2, solver=solver, non_negative=true, adding_up=true)

    # 3. Solve for time weights
    time_target = mean(O[is_control, (Tpre + 1):end];
                       dims=2) |> vec
    λ0, λpre = constrained_elastic_net(
        Opre, time_target;
        solver=solver, non_negative=true, adding_up=true)

    # 4. Solve the weighted DID regression
    ω = zeros(N)
    ω[is_control] .= ωco
    ω[.!is_control] .= 1 / Ntr
    λ = vcat(λpre, ones(Tpost) ./ Tpost)

    m = Model(solver)
    @variable(m, α[1:N])
    @variable(m, β[1:T])
    @variable(m, τ)
    @objective(m, Min, sum((O .- α .- β' .- τ .* Z) .^ 2 .* (ω * λ')))
    optimize!(m)
    value(τ)
    # step = partial(alternating_minimization_step, O, Z, ω * λ')
    # init = (0., zeros(N, 1), zeros(1, T))
    # τ, α, β = step_to_tol(step, init; kwargs...)
    # τ
end

# function convex_linear_regression(X::Array{S, 2}, y::Array{S, 1};
#                                   ρ=0., solver=DEFAULT_SOLVER)  where S <: Real
#     """
#     min_{β0, β} (1/n)|| y - X β - β0 ||^2_2 + ρ ||β||^2_2
#     s.t. β ≥ 0, sum(β) == 1
#     """
#     d = size(X, 2)
#     m = Model(solver)
#     @variable(m, β0)
#     @variable(m, β[1:d] >= 0)
#     @objective(m, Min, mean((β0 .+ X * β .- y) .^ 2) + ρ * β'β)
#     @constraint(m, sum(β) == 1)
#     optimize!(m)
#     value(β0), max.(value.(β), 0.)
# end

function alternating_minimization_step(O, Z, W, params)
    τ, α, β = params
    αnew = wmean(O .- β .- τ * Z, W, 2)
    βnew = wmean(O .- α .- τ * Z, W, 1)
    τnew = wmean(Z .* (O - αnew * βnew), Z .* W)
    (τnew, αnew, βnew)
end
