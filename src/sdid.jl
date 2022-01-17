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
end
