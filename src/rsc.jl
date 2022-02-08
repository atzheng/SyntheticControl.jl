function SVT(X; μ=0., r=Inf)
    U, S, V = svd(X)
    soft_r = sum(S .>= μ)
    r = Int(min(soft_r, r))
    U[:, 1:r], S[1:r], V[:, 1:r]
end

reconstruct(U, S, V) = U * diagm(S) * V'

function RSC(O::Array{S, 2}, Z::Array{S, 2};
             λ=0., μ=0., r=Inf, Tpre=nothing,
             solver=Ipopt.Optimizer) where S <: Real
    is_control, Tpre_inferred = infer_controls(Z)
    Tpre = Tpre !== nothing ? Tpre : Tpre_inferred
    U, Σ, V = SVT(O[is_control, :]; μ=μ, r=r)
    Mhat = reconstruct(U, Σ, V)
    Y1pre = vec(mean(O[.!is_control, 1:Tpre]; dims=1))
    _, β = constrained_elastic_net(
        collect(Mhat[:, 1:Tpre]'), Y1pre;
        α=0., λ=λ, intercept=false, solver=solver)

    # Get timesteps where all treatment units were treated
    Ttreat = vec(all(Z[.!is_control, :] .== 1; dims=1))
    Y1post = vec(mean(O[.!is_control, Ttreat]; dims=1))
    M1hatpost = vec(Mhat[:, Ttreat]'β)
    mean(Y1post - M1hatpost)
end
