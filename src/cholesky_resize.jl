"""
    delete_column!(F::UpdatableCholesky, j) -> F

Remove index `j`, deleting both row `j` and column `j` of the factored matrix, in `O(n^2)`
operations.

Writing the factor in blocks about `j`, the leading block and the block below and to the left of
`j` are already the factors of the reduced matrix. Only the trailing block changes, and it
changes by a rank-1 update with the deleted column's tail.

Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5.
"""
function delete_column!(F::UpdatableCholesky, j::Integer)
    n = F.n
    1 <= j <= n || throw(BoundsError(F, j))
    L = _lower(F)
    tail = [L[i, j] for i in (j + 1):n]
    for c in 1:(n - 1)
        for r in 1:(n - 1)
            L[r, c] = L[r + (r >= j), c + (c >= j)]
        end
    end
    for k in 1:n                       # clear the vacated last row and column
        L[n, k] = zero(eltype(L))
        L[k, n] = zero(eltype(L))
    end
    F.n = n - 1
    isempty(tail) || _ch1up!(view(_lower(F), j:(n - 1), j:(n - 1)), tail)
    return F
end

function _grow!(F::UpdatableCholesky{T}, needed::Int) where {T}
    needed <= capacity(F) && return F
    newcap = max(needed, 2 * capacity(F))
    f = zeros(T, newcap, newcap)
    copyto!(view(f, 1:F.n, 1:F.n), view(F.factors, 1:F.n, 1:F.n))
    F.factors = f
    resize!(F.work, newcap)
    resize!(F.cosines, newcap)
    resize!(F.rot, newcap)
    return F
end

# Append a new last index. `x` has length n+1, with x[n+1] the new diagonal entry.
function _append!(F::UpdatableCholesky{T}, x::AbstractVector) where {T}
    n = F.n
    length(x) == n + 1 ||
        throw(DimensionMismatch("x has length $(length(x)), expected $(n + 1)"))
    L = _lower(F)
    l = Vector{T}(undef, n)
    for i in 1:n
        acc = x[i]
        for k in 1:(i - 1)
            acc -= L[i, k] * l[k]
        end
        l[i] = acc / L[i, i]
    end
    d = real(x[n + 1]) - sum(abs2, l)
    d > 0 || throw(PosDefException(n + 1))
    _grow!(F, n + 1)
    F.n = n + 1
    M = _lower(F)
    for k in 1:n
        M[n + 1, k] = conj(l[k])
        M[k, n + 1] = zero(T)
    end
    M[n + 1, n + 1] = sqrt(d)
    return F
end
