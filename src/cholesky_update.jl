"""
    lowrankupdate!(F::UpdatableCholesky, v) -> F

Replace the factorization of `A` with that of `A + v*v'` in `O(n^2)` operations. `v` is not
modified. Extends `LinearAlgebra.lowrankupdate!`.

Gill, Golub, Murray and Saunders, *Methods for modifying matrix factorizations*,
Mathematics of Computation 28 (1974), 505-535.
"""
function LinearAlgebra.lowrankupdate!(F::UpdatableCholesky, v::AbstractVector)
    length(v) == F.n ||
        throw(DimensionMismatch("v has length $(length(v)), factorization is $(F.n)"))
    w = view(F.work, 1:F.n)
    copyto!(w, v)
    _ch1up!(_lower(F), w)
    return F
end

# Chase a rank-1 term into a lower-triangular factor with Givens rotations. `w` is consumed.
function _ch1up!(L, w)
    n = size(L, 1)
    for k in 1:n
        c, s, r = givensAlgorithm(L[k, k], w[k])
        L[k, k] = r
        for i in (k + 1):n
            Lik = L[i, k]
            wi = w[i]
            L[i, k] = c * Lik + s * wi
            w[i] = -conj(s) * Lik + c * wi
        end
    end
    return L
end
