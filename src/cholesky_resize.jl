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
    tail = view(F.work, 1:(n - j))
    for i in (j + 1):n
        tail[i - j] = L[i, j]
    end
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
    resize!(F.perm, newcap)
    return F
end

# Append a new last index. `x` has length n+1, with its last entry the new diagonal entry.
function _append!(F::UpdatableCholesky{T}, x::AbstractVector) where {T}
    n = F.n
    length(x) == n + 1 ||
        throw(DimensionMismatch("x has length $(length(x)), expected $(n + 1)"))
    L = _lower(F)
    l = view(F.work, 1:n)
    ix = firstindex(x) - 1
    for i in 1:n
        l[i] = x[ix + i]
    end
    # Solves L l = l in place. `L` is strided, so a `BlasFloat` element type dispatches to
    # `trtrs`; every other element type falls back to a column-oriented generic loop, which
    # reads `L[i, k]` with `k` fixed -- contiguous in `i` -- rather than striding across the
    # capacity-sized storage `L` is a view into.
    ldiv!(LowerTriangular(L), l)
    d = real(x[ix + n + 1]) - sum(abs2, l)
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

# Swap adjacent indices k, k+1 of the factored matrix, restoring lower-triangular form.
# Swapping rows k and k+1 of L disturbs only entry (k, k+1): row k+1 held the only nonzero
# entry at column k+1, so it becomes a single superdiagonal entry in row k. One Givens
# rotation mixing columns k and k+1, applied to every row from k down to n, eliminates it and
# leaves L lower triangular again. Chaining this over the |i - j| steps between two indices
# costs O(n) per step, hence O((|i - j| + 1) * n) overall, rather than a full
# re-triangularization of the whole factor.
function _adjacent_shift!(F::UpdatableCholesky, k::Int)
    n = F.n
    L = _lower(F)
    for c in 1:(k - 1)
        L[k, c], L[k + 1, c] = L[k + 1, c], L[k, c]
    end
    akk, ak1k, bulge = L[k, k], L[k + 1, k], L[k + 1, k + 1]
    L[k, k] = ak1k
    L[k + 1, k] = akk
    L[k, k + 1] = bulge
    L[k + 1, k + 1] = zero(eltype(L))
    cc, ss, rr = givensAlgorithm(L[k, k], L[k, k + 1])
    L[k, k] = rr
    L[k, k + 1] = zero(eltype(L))
    for i in (k + 1):n
        a = L[i, k]
        b = L[i, k + 1]
        L[i, k] = cc * a + ss * b
        L[i, k + 1] = -conj(ss) * a + cc * b
    end
    return F
end

# A Cholesky factor is unique only up to a unit-modulus scaling of each column, and the
# rotations in `_adjacent_shift!` do not constrain that scaling. Fix it, over the columns
# `_adjacent_shift!` touched, so the diagonal is real and positive: scaling column k by
# s = conj(L[k,k])/abs(L[k,k]) leaves L*L' unchanged and makes the diagonal entry abs(L[k,k]),
# which is written directly so that it is exactly real.
function _fixdiag!(L, lo::Int, hi::Int)
    n = size(L, 1)
    for k in lo:hi
        dkk = L[k, k]
        iszero(dkk) && continue
        s = conj(dkk) / abs(dkk)
        if !isone(s)
            for i in (k + 1):n
                L[i, k] *= s
            end
        end
        L[k, k] = abs(dkk)
    end
    return L
end

# Fill `p` with the permutation that moves the index at position `i` to position `j`, sliding
# the indices between them by one.
function _cyclicperm!(p::AbstractVector{Int}, i::Int, j::Int)
    for k in eachindex(p)
        p[k] = k
    end
    if i <= j
        for k in i:(j - 1)
            p[k] = k + 1
        end
    else
        for k in (j + 1):i
            p[k] = k - 1
        end
    end
    p[j] = i
    return p
end

"""
    shift_columns!(F::UpdatableCholesky, i, j) -> F

Move index `i` to position `j`, sliding the indices between them by one, and update the
factorization to match, in `O((|i - j| + 1) * n)` operations. Both the row and the column
move, keeping the factored matrix symmetric.

Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5.
"""
function shift_columns!(F::UpdatableCholesky, i::Integer, j::Integer)
    1 <= i <= F.n || throw(BoundsError(F, i))
    1 <= j <= F.n || throw(BoundsError(F, j))
    ii, jj = Int(i), Int(j)
    if ii < jj
        for k in ii:(jj - 1)
            _adjacent_shift!(F, k)
        end
    elseif ii > jj
        for k in (ii - 1):-1:jj
            _adjacent_shift!(F, k)
        end
    end
    _fixdiag!(_lower(F), min(ii, jj), max(ii, jj))
    return F
end

"""
    insert_column!(F::UpdatableCholesky, j, x) -> F

Insert a new index at position `j`, adding both a row and a column. `x` is the new row and
column in the resulting indexing, so it has length `n+1` and `x[j]` is the new diagonal entry.

The new index is appended at position `n+1` and then moved down to `j` with `shift_columns!`, in
`O((n - j + 2) * n)` operations: inserting near `n+1` is cheap, and inserting near `1` costs as
much as `shift_columns!(F, 1, n)`.

Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating the
Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772-795.
"""
function insert_column!(F::UpdatableCholesky, j::Integer, x::AbstractVector)
    n = F.n
    length(x) == n + 1 ||
        throw(DimensionMismatch("x has length $(length(x)), expected $(n + 1)"))
    1 <= j <= n + 1 || throw(BoundsError(F, j))
    # Growing here leaves _append!'s own growth a no-op, so `y` stays a view of live storage.
    _grow!(F, n + 1)
    y = view(F.rot, 1:(n + 1))
    ix = firstindex(x) - 1
    for k in 1:(j - 1)
        y[k] = x[ix + k]
    end
    for k in (j + 1):(n + 1)
        y[k - 1] = x[ix + k]
    end
    y[n + 1] = x[ix + j]
    _append!(F, y)
    return j == n + 1 ? F : shift_columns!(F, n + 1, Int(j))
end
