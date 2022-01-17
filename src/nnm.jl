function estimate_treatment(
    O::Array{<: Real, 2}, Z::Array{<: Integer, 2}, r::Integer; kwargs...)
    # if !is_valid_input(O, Z, r) error("Invalid input.") end
    τ, M, λ = NNM(O, Z, r; kwargs...)
    τ, asymptotic_error_distribution(M, Z, λ)
end

function NNM(O::Array{T, 2}, Z::Array{<: Integer, 2};
             τ0=0., λ=1., kwargs...) where T <: Real
    """
    Proximal gradient descent for regularized NNM:
    min_{M, τ} || O - τ Z - M ||_F^2 + λ ||M||_*^2
    """
    step = partial(convex_estimator_step, O, Z, λ)
    τ = step_to_tol(step, τ0; kwargs...)
    M = soft_threshold(O - τ .* Z, λ)
    τ, M
end

function NNM(O::Array{T, 2}, Z::Array{<: Integer, 2}, r::Integer;
             τ0=0., λmax=nothing, λmin=1e-3, coef=1.05, kwargs...) where T <: Real
    "Tunes λ to achieve rank r."
    m, n = size(O)
    τ = τ0
    M = nothing
    λmax = if λmax === nothing svd(O - τ0 .* Z).S[2] else λmax end
    n_λs = (log(λmax) - log(λmin)) / log(coef)
    check_rank(x1, x2) = if rank(x2[2]) > r reduced(x1) else x2 end
    check_rank(x1::Nothing, x2) = x2
    step(τ, M, λ) =
        (NNM(O, Z; λ=λ / coef, τ0=τ, kwargs...)..., λ / coef)
    M0 = O - τ0 .* Z
    result = transduce(Iterated(splat(step), (τ0, M0, λmax * coef)),
                       Completing(check_rank), nothing, 1:n_λs)
    result |> ifunreduced() do x
        println("Warning: rank r not achieved.")
        nothing
    end
end

function convex_estimator_step(
    O::Array{T, 2}, Z::Array{<: Integer, 2}, λ::T, τ::T) where T <: Real
    M = soft_threshold(O - τ .* Z, λ)
    sum(Z .* (O - M)) / sum(Z)
end

function soft_threshold(M::Array{T, 2}, λ::T) where T <: Real
    U, S, V = svd(M)
    Sp = max.(S .- λ, 0)
    U[:, Sp .>= 0] .*
        reshape(Sp[Sp .>= 0], (1, sum(Sp .>= 0))) *
        V[:, Sp .>= 0]'
end

function asymptotic_error_distribution(Mhat::Array{T, 2},
                                       Z::Array{<: Integer, 2},
                                       λ::Real; σ=1.) where T <: Real
    U, S, V = svd(Mhat)
    PTperpZ = (I - U * U') * (Z - Z * V * V')
    PTperpZ_norm2 = sum(PTperpZ .^ 2)
    μ = λ * sum(Z .* (U * V')) / PTperpZ_norm2
    # Assuming iid error with sd σ.
    # TODO generalize to independent error
    σasy = σ / sqrt(PTperpZ_norm2)
    Normal(μ, σasy)
end

# @memoize LRU{Tuple{Array{Float64, 2}, Int64, Bool},
#              SVD{Float64, Float64, Array{Float64, 2}}}(
#                  maxsize=1) function svt(M; r=rank(M), randomized=false)
#     m, n = size(M)
#     if randomized && m >= r && n >= r
#         # Need this logic due to bugs in rsvd_fnkz
#         try
#             if m >= n
#                 rsvd_fnkz(M, r)
#             else
#                 result = rsvd_fnkz(M', r)
#                 SVD(collect(result.V), result.S, collect(result.U'))
#             end
#         catch e
#             println(e)
#             println(@sprintf("Randomized SVD failed at time %d", s.t))
#             svt(M; r=r, randomized=false)
#         end
#     else
#         U, S, V = svd(M)
#         SVD(U[:, 1:r], S[1:r], collect(V[:, 1:r]'))
#     end
# end
