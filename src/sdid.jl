using StatsBase


function SDID(O::Array{S, 2}, Z::Array{S, 2};
              Tpre=nothing, solver=Ipopt.Optimizer,
              verbose=false,
              non_negative=true,
              adding_up=true,
              kwargs...) where S <: Real
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
    if verbose println("Solving for unit weights...") end
    Opre = O[is_control, 1:Tpre]
    ω0, ωco = constrained_elastic_net(
        collect(Opre'),
        vec(mean(O[.!is_control, 1:Tpre]; dims=1));
        λ=ζ2, α=0., solver=solver,
        non_negative=non_negative, adding_up=adding_up,
        verbose=verbose)

    # 3. Solve for time weights
    if verbose println("Solving for time weights...") end
    time_target = mean(O[is_control, (Tpre + 1):end];
                       dims=2) |> vec
    λ0, λpre = constrained_elastic_net(
        Opre, time_target;
        solver=solver, non_negative=non_negative,
        adding_up=adding_up, verbose=verbose)

    # 4. Solve the weighted DID regression
    if verbose println("Solving weighted regression...") end
    ω = zeros(N)
    ω[is_control] .= ωco
    ω[.!is_control] .= 1 / Ntr
    λ = vcat(λpre, ones(Tpost) ./ Tpost)

    m = Model(solver)
    @variable(m, α[1:N])
    @variable(m, β[1:T])
    @variable(m, τ)
    @objective(m, Min, sum((O .- α .- β' .- τ .* Z) .^ 2 .* (ω * λ')))
    if !verbose set_silent(m) end
    optimize!(m)
    value(τ)
    # step = partial(alternating_minimization_step, O, Z, ω * λ')
    # init = (0., zeros(N, 1), zeros(1, T))
    # τ, α, β = step_to_tol(step, init; kwargs...)
    # τ
end

function alternating_minimization_step(O, Z, W, params)
    τ, α, β = params
    αnew = wmean(O .- β .- τ * Z, W, 2)
    βnew = wmean(O .- α .- τ * Z, W, 1)
    τnew = wmean(Z .* (O .- αnew .-  βnew), Z .* W)
    (τnew, αnew, βnew)
end
