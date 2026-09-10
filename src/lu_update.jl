"""
    lowrankupdate!(F::UpdatableLU, u, v; rtol = 0) -> F

Replace the factorization of `A` with that of `A + u*v'` in `O(n^2)` operations. Neither `u` nor
`v` is modified. The permutation is retained rather than recomputed, so a pivot that becomes
small is not repaired.

`ZeroPivotException` is thrown when a pivot reaches zero, and, when `rtol > 0`, also when a
pivot falls below `rtol` times the magnitude of the two terms that formed it. The error in the
updated factors grows like `|d_k| / |d'_k|`, the ratio of a pivot before and after the update,
so a loop of updates should set `rtol` rather than rely on the zero test alone.

A throw leaves the factorization invalid; `issuccess(F)` is `false` afterwards and `F` must be
rebuilt.

Bennett, *Triangular factors of modified matrices*, Numerische Mathematik 7 (1965), 217-221.
The pivoted case follows Stange, Griewank and Bollhöfer, *On the efficient update of rectangular
LU-factorizations subject to low rank modifications*, ETNA 26 (2007), 161-177, in applying the
update in the permuted frame.
"""
function LinearAlgebra.lowrankupdate!(
        F::UpdatableLU{T}, u::AbstractVector,
        v::AbstractVector; rtol::Real = 0
    ) where {T}
    _checkvalid(F)
    n = length(getfield(F, :d))
    length(u) == n || throw(DimensionMismatch("u has length $(length(u)), factorization is $n"))
    length(v) == n || throw(DimensionMismatch("v has length $(length(v)), factorization is $n"))
    p = getfield(F, :p)
    work = getfield(F, :work)
    w = view(work, 1:n)
    z = view(work, (n + 1):2n)
    iu = firstindex(u) - 1
    iv = firstindex(v) - 1
    for i in 1:n
        w[i] = u[iu + p[i]]
    end
    # `_bennett!` works in the transpose form, so A + u*v' is passed as second vector conj(v).
    # Copy elementwise: `conj(v)` returns `v` itself for real `v` -- Base defines
    # conj(::AbstractArray{<:Real}) = v -- and the kernel consumes what it is given.
    for i in 1:n
        z[i] = conj(v[iv + i])
    end
    k = @strict _bennett!(getfield(F, :Lf), getfield(F, :d), getfield(F, :Ut), w, z, one(T), rtol)
    if !iszero(k)
        setfield!(F, :info, k)
        throw(ZeroPivotException(k))
    end
    return F
end

# A + sigma*w*transpose(z), on the LDU form. `w` and `z` are consumed. Returns zero, or the
# column at which the pivot became too small to continue, leaving the factors partly overwritten.
# `U` is the transpose of the unit upper triangular factor (`U[j, k] == U-factor[k, j]`), so both
# loops below walk down a column -- contiguous in a column-major array, like `L`'s own layout.
function _bennett!(L, d, U, w, z, sigma, rtol)
    n = length(d)
    s = sigma
    for k in 1:n
        wk = w[k]
        zk = z[k]
        term = s * wk * zk
        dnew = d[k] + term
        (iszero(dnew) || abs(dnew) < rtol * (abs(d[k]) + abs(term))) && return k
        alpha = s * zk / dnew
        beta = s * wk / dnew
        for i in (k + 1):n
            w[i] -= wk * L[i, k]
            L[i, k] += alpha * w[i]
        end
        for j in (k + 1):n
            z[j] -= zk * U[j, k]
            U[j, k] += beta * z[j]
        end
        s = s * d[k] / dnew
        d[k] = dnew
    end
    return 0
end
