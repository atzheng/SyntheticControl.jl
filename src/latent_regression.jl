using Printf


function LatentRegression(
    O::Array{S, 2}, Z::Array{S, 2};
    r=size(O, 1), randomized=false,
    kwargs...
) where S <: Real
    is_control, Tpre_inferred = infer_controls(Z)
    U, _, V = svt(O[is_control, :]; r=r, randomized=randomized)
    @assert is_control[1] == false
    X1 = Z[1, :]
    X = hcat(X1, V)
    Utr_est = (X'X) \ X' * O[1, :]
    Utr_est[1]
end


function svt(M; r=rank(M), randomized=false)
    m, n = size(M)
    if randomized && m >= r && n >= r
        # Need this logic due to bugs in rsvd_fnkz
        try
            if m >= n
                rsvd_fnkz(M, r)
            else
                result = rsvd_fnkz(M', r)
                SVD(collect(result.V), result.S, collect(result.U'))
            end
        catch e
            println(e)
            svt(M; r=r, randomized=false)
        end
    else
        U, S, V = svd(M)
        SVD(U[:, 1:r], S[1:r], collect(V[:, 1:r]'))
    end
end
