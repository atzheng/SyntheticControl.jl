is_tol(x::Real, y::Real, tol::Real) = abs(x - y) < tol
is_tol(x::Array, y::Array, tol::Real) = norm(x - y, 2) < tol
is_tol(x::Tuple, y::Tuple, tol::Real) =
    all(is_tol(a, b, tol) for (a, b) in zip(x, y))

function check_tol(τ1, τ2; tol=1e-3)
    if τ1 != nothing && is_tol(τ1, τ2, tol)
        reduced(τ2)
    else
        @show τ2[1]
        τ2
    end
end

function step_to_tol(stepper, init; max_iters=1000, tol=1e-5)
    """
    A generic implementation for iterative algorithms with early stopping
    based on convergence checking.
    """
    result = transduce(Iterated(stepper, init),
                       Completing(partial(check_tol; tol=tol)),
                       nothing, 1:max_iters)
    if !(result isa Reduced) @warn("Max iters reached.") end
    unreduced(result)
end

wmean(X::Array{T, 2}, W::Array{T, 2}, dims::Int) where T <: Real =
    sum(X .* W; dims=dims) ./ sum(W; dims=dims)

wmean(X, W) = sum(X .* W) ./ sum(W)

function infer_controls(Z)
    """
    Guesses the pre-treatment period and control units from the treatment
    matrix Z.
    """
    is_control = vec(sum(Z; dims=2)) .== 0
    Tpre = minimum([x[2] for x in findmax(Z[.!is_control, :]; dims=2)[2]])
    is_control, Tpre
end
