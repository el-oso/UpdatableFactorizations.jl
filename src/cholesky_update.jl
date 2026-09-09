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
    iv = firstindex(v) - 1
    for i in 1:F.n
        w[i] = v[iv + i]
    end
    @strict _ch1up!(_lower(F), w)
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

"""
    lowrankdowndate!(F::UpdatableCholesky, v) -> F

Replace the factorization of `A` with that of `A - v*v'` in `O(n^2)` operations. `v` is not
modified. Throws `PosDefException` when the result would not be positive definite. Extends
`LinearAlgebra.lowrankdowndate!`.

Uses the mixed formulation of Bojanczyk, Brent, Van Dooren and de Hoog, *A note on downdating
the Cholesky factorization*, SIAM Journal on Scientific and Statistical Computing 8 (1987),
210-221, which is more stable near breakdown than forming the hyperbolic rotation directly.
"""
function LinearAlgebra.lowrankdowndate!(F::UpdatableCholesky, v::AbstractVector)
    length(v) == F.n ||
        throw(DimensionMismatch("v has length $(length(v)), factorization is $(F.n)"))
    w = view(F.work, 1:F.n)
    iv = firstindex(v) - 1
    for i in 1:F.n
        w[i] = v[iv + i]
    end
    @strict _ch1dn!(_lower(F), w, view(F.cosines, 1:F.n), view(F.rot, 1:F.n))
    return F
end

# `w` is consumed, first as the right-hand side of L p = w and then as p itself. `cs` and `sn`
# are scratch of length n; taking them from the caller keeps this allocation-free.
function _ch1dn!(L, w, cs, sn)
    n = size(L, 1)
    T = eltype(L)
    R = real(T)
    p = w
    for i in 1:n
        acc = p[i]
        for k in 1:(i - 1)
            acc -= L[i, k] * p[k]
        end
        p[i] = acc / L[i, i]
    end
    alpha = one(R) - sum(abs2, p)
    alpha > 0 || throw(PosDefException(n))
    a = sqrt(alpha)
    for i in n:-1:1
        r = hypot(a, abs(p[i]))
        cs[i] = a / r
        sn[i] = p[i] / r
        a = r
    end
    for j in 1:n
        xx = zero(T)
        for i in j:-1:1
            t = cs[i] * xx + sn[i] * L[j, i]
            L[j, i] = cs[i] * L[j, i] - conj(sn[i]) * xx
            xx = t
        end
    end
    return L
end
