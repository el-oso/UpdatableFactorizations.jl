# Zero every subdiagonal entry of `Rv` with row rotations, mirrored onto `q`'s columns so that
# Q*R is unchanged. Entries that are already zero are skipped, so a disturbance confined to a
# band costs only that band. Column-major order with the rows taken from the bottom up covers
# both shapes that arise: a Hessenberg subdiagonal, and a single spike in one column.
function _retriangularize!(Rv::AbstractMatrix{T}, q::AbstractQRep{T}) where {T}
    nr, nc = size(Rv)
    for c in 1:nc, r in min(nr, nc + 1):-1:(c + 1)
        iszero(Rv[r, c]) && continue
        cc, ss, rr = givensAlgorithm(Rv[r - 1, c], Rv[r, c])
        G = Givens(r - 1, r, oftype(Rv[r, c], cc), oftype(Rv[r, c], ss))
        lmul!(G, Rv)
        rmul!(q, G')
        Rv[r - 1, c] = rr
        Rv[r, c] = zero(T)    # lmul! leaves a rounding residue; the invariant is an exact zero
    end
    return Rv
end

"""
    delete_column!(F::UpdatableQR, j) -> F

Remove column `j` of the factored matrix, in `O((n - j)(m + n))` operations.

Sliding the trailing columns of `R` one place to the left leaves it upper Hessenberg from column
`j` on, and rotations chase that subdiagonal back to zero. The columns before `j` are untouched,
so an early deletion costs the whole factor and a late one costs almost nothing.

Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5.
"""
function delete_column!(F::UpdatableQR{T, S, <:DenseQ}, j::Integer) where {T, S}
    n = F.n
    1 <= j <= n || throw(BoundsError(F, j))
    q = getfield(F, :qrep)
    R = getfield(F, :factors)
    for c in j:(n - 1), r in 1:n
        R[r, c] = R[r, c + 1]
    end
    n == 1 || _retriangularize!(view(R, 1:n, 1:(n - 1)), q)
    _dropcolumn!(q)
    F.n = n - 1
    # Re-establish zero storage outside the new active block, mirroring what `_dropcolumn!`
    # does for Q: nothing above assumes the vacated column or the trailing region was already
    # zero.
    fill!(view(R, (F.n + 1):size(R, 1), :), zero(T))
    fill!(view(R, :, (F.n + 1):size(R, 2)), zero(T))
    return F
end

"""
    shift_columns!(F::UpdatableQR, i, j) -> F

Move column `i` of the factored matrix to position `j`, sliding the columns between them by one,
in `O(|i - j|(m + n))` operations.

Moving a column right leaves `R` upper Hessenberg over the columns it passed; moving it left
leaves a single spike in column `j`. Rotations clear both.

Reichel and Gragg, *Algorithm 686: FORTRAN subroutines for updating the QR decomposition*,
ACM Transactions on Mathematical Software 16 (1990), 369-377.
"""
function shift_columns!(F::UpdatableQR{T, S, <:DenseQ}, i::Integer, j::Integer) where {T, S}
    n = F.n
    1 <= i <= n || throw(BoundsError(F, i))
    1 <= j <= n || throw(BoundsError(F, j))
    q = getfield(F, :qrep)
    R = getfield(F, :factors)
    if i != j
        hold = _rspare(F)
        for r in 1:n
            hold[r] = R[r, i]
        end
        if i < j
            for c in i:(j - 1), r in 1:n
                R[r, c] = R[r, c + 1]
            end
        else
            for c in i:-1:(j + 1), r in 1:n
                R[r, c] = R[r, c - 1]
            end
        end
        for r in 1:n
            R[r, j] = hold[r]
            hold[r] = zero(T)
        end
        _retriangularize!(view(R, 1:n, 1:n), q)
    end
    # Re-establish zero storage outside the active block: `n` does not change here, so nothing
    # above assumes the spare column, spare row, or Q's trailing columns were already zero, and
    # the invariant must hold whether or not `i == j` skipped the shift itself.
    fill!(view(R, (n + 1):size(R, 1), :), zero(T))
    fill!(view(R, :, (n + 1):size(R, 2)), zero(T))
    fill!(view(q.buf, :, (n + 1):size(q.buf, 2)), zero(T))
    return F
end

"""
    delete_row!(F::UpdatableQR, i; rtol = sqrt(eps(real(T)))) -> F

Remove row `i` of the factored matrix, in `O(mn)` operations.

The part of `e_i` orthogonal to the range of `Q` augments the factor to `m x (n+1)` columns, and
`n` rotations move row `i` of that augmented factor onto its last column, which is then dropped
along with the row. `gamma`, the norm of that orthogonal part, is the distance of `e_i` from the
range of `Q`: it lies in `[0, 1]`, and it agrees with the smallest singular value of the
remaining rows to several digits. `ArgumentError` is thrown, naming the row and `gamma`, when
`gamma <= rtol`; the deleted row then has leverage one and the numerical rank of what remains has
dropped, so the factorization that would be returned is a factorization of something else.

`gamma` is computed as the norm of the residual rather than from `sqrt(1 - norm(Q'e_i)^2)`. The
algebraic form subtracts nearly equal quantities: it reaches exactly zero for rows whose true
`gamma` is around `1e-15`, and goes negative around `1e-9`, so it refuses accurate deletions and
admits destructive ones.

`DimensionMismatch` is thrown when the factorization is square, because the type admits only
`m >= n`.

Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating the
Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772-795.
Reichel and Gragg, *Algorithm 686: FORTRAN subroutines for updating the QR decomposition*,
ACM Transactions on Mathematical Software 16 (1990), 369-377.
"""
function delete_row!(
        F::UpdatableQR{T, S, <:DenseQ}, i::Integer; rtol::Real = sqrt(eps(real(T)))
    ) where {T, S}
    m, n = F.m, F.n
    1 <= i <= m || throw(BoundsError(F, i))
    m > n || throw(
        DimensionMismatch(
            "deleting row $i would leave a $(m - 1)x$n factorization; " *
                "the factorization requires m >= n"
        )
    )
    q = getfield(F, :qrep)
    Qa = _active(q)
    t = view(F.work, 1:n)
    corr = view(F.corr, 1:n)
    z = _spare(q)
    for k in 1:n
        t[k] = conj(Qa[i, k])
    end
    mul!(z, Qa, t, -one(T), zero(T))
    z[i] += one(T)
    g = _project_residual!(t, z, Qa, corr)
    if !(g > rtol)
        # The projection built the augmentation direction in the spare column. Clearing it is
        # what leaves the factorization as it was, and the next verb builds there.
        _clearspare!(q)
        throw(
            ArgumentError(
                "row $i has leverage one: gamma = $g; deleting it drops the numerical rank"
            )
        )
    end
    z ./= g
    QA = _augmented(q)
    RA = _raug(F)
    for k in n:-1:1
        # Rotating row i of [Q z] on the right by G' sends its k-th entry to c*a + conj(s)*b,
        # with a = QA[i,k] and b = QA[i,n+1]; givensAlgorithm(-b, a) is the pair that makes it
        # vanish. Descending k keeps R upper triangular, so it is not retriangularized.
        a = QA[i, k]
        b = QA[i, n + 1]
        c, s, _ = givensAlgorithm(-b, a)
        G = Givens(k, n + 1, oftype(a, c), oftype(a, s))
        lmul!(G, RA)
        rmul!(q, G')
        QA[i, k] = zero(T)
    end
    _deleterow!(q, Int(i))
    F.m = m - 1
    # Re-establish zero storage outside the active block: row n+1 of R held the coefficients of
    # the deleted row as working space, and nothing above assumes it, R's spare columns, or Q's
    # spare columns were already zero on entry.
    Rfull = getfield(F, :factors)
    fill!(view(Rfull, (n + 1):size(Rfull, 1), :), zero(T))
    fill!(view(Rfull, :, (n + 1):size(Rfull, 2)), zero(T))
    fill!(view(q.buf, :, (n + 1):size(q.buf, 2)), zero(T))
    return F
end

# Project `r` onto the columns of `Qa`, leaving the coefficients in `w` and the residual in `r`,
# and return the residual norm. The second pass is unconditional: with one pass the loss of
# orthogonality grows like eps divided by the residual norm, which is unbounded as the residual
# collapses, while two passes hold it at O(eps).
#
# Giraud, Langou and Rozlozník, *The loss of orthogonality in the Gram-Schmidt orthogonalization
# process*, Computers and Mathematics with Applications 50 (2005), 1069-1075.
function _project!(w, r, Qa, corr)
    mul!(w, Qa', r)
    mul!(r, Qa, w, -1, true)
    return _project_residual!(w, r, Qa, corr)
end

# The second pass alone, for a caller that has already applied the first and accumulated its
# coefficients in `w`.
function _project_residual!(w, r, Qa, corr)
    mul!(corr, Qa', r)
    mul!(r, Qa, corr, -1, true)
    w .+= corr
    return norm(r)
end

# A destination for `_absorb_spike!`'s rotations that discards them: holding no data, it costs
# nothing to construct or to rotate, so a caller that only wants the resulting `RA` mirrors
# onto this instead of a real `Q`.
struct _NoQ{T} <: AbstractQRep{T} end
LinearAlgebra.rmul!(::_NoQ, ::Givens) = nothing

# Chase the spike `z` into `RA` with Givens rotations, add the rank-1 correction to row 1 of
# `RA`, and retriangularize, mirroring every rotation onto `q`. The sequence of rotations
# depends only on `z` and `RA`, never on `q`'s entries, so calling this once on a scratch `RA`,
# `z` and `q` and again on the live ones produces identical numbers in both `RA`s.
function _absorb_spike!(RA, z, q, v, iv, n, last)
    T = eltype(z)
    for k in (last - 1):-1:1
        c, s, rr = givensAlgorithm(z[k], z[k + 1])
        G = Givens(k, k + 1, oftype(z[k], c), oftype(z[k], s))
        z[k] = rr
        z[k + 1] = zero(T)
        lmul!(G, RA)
        rmul!(q, G')
    end
    for j in 1:n
        RA[1, j] += z[1] * conj(v[iv + j])
    end
    _retriangularize!(RA, q)
    return RA
end

"""
    lowrankupdate!(F::UpdatableQR, u, v; rtol = sqrt(eps(real(T)))) -> F

Replace the factorization of `A` with that of `A + u*v'` in `O(mn)` operations. Neither `u` nor
`v` is modified.

`u` is split into its projection onto the range of `Q` and a residual. When the residual is
negligible relative to `u`, or when the factorization is square and so has no room for a new
direction, the update is carried out inside the existing range; that is a legitimate case rather
than a failure, and the routine never throws for it.

Because `Q` is orthonormal, the norm of column `j` of the updated `R` equals the norm of column
`j` of `A + u*v'` itself, so `abs(R[j,j]) / norm(R[1:j,j])` is a dimensionless ratio in `[0, 1]`
for every column: it is `1` when column `j` carries no contribution from the columns before it,
and it collapses toward `0` as `u*v'` drives column `j` into their span. `ArgumentError` is
thrown, naming the column, when this ratio is at or below `rtol`: a column driven into the span
of the others leaves the factorization exact but its diagonal entry at the level of rounding
noise, so a later solve through it divides by that noise. `rtol = 0` disables this check,
admitting any update including an exactly singular one.

When `rtol` is nonzero, the candidate `R` is computed in `F`'s own scratch storage, and the
check runs against it, before `Q` or the stored factor are touched; a thrown update therefore
leaves the factorization exactly as it was. This call allocates nothing, at any `rtol`.

Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating the
Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772-795.
Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5.
"""
function LinearAlgebra.lowrankupdate!(
        F::UpdatableQR{T, S, <:DenseQ}, u::AbstractVector, v::AbstractVector;
        rtol::Real = sqrt(eps(real(T)))
    ) where {T, S}
    m, n = F.m, F.n
    length(u) == m ||
        throw(DimensionMismatch("u has length $(length(u)), factorization is $(m)x$(n)"))
    length(v) == n ||
        throw(DimensionMismatch("v has length $(length(v)), factorization is $(m)x$(n)"))
    q = getfield(F, :qrep)
    Qa = _active(q)
    r = _spare(q)
    z = view(F.work, 1:(n + 1))
    w = view(z, 1:n)
    corr = view(F.corr, 1:n)
    iu = firstindex(u) - 1
    for i in 1:m
        r[i] = u[iu + i]
    end
    unrm = norm(r)
    rho = _project!(w, r, Qa, corr)
    RA = _raug(F)
    if m > n && rho > n * eps(real(T)) * unrm
        r ./= rho
        z[n + 1] = rho
        last = n + 1
    else
        fill!(r, zero(T))
        z[n + 1] = zero(T)
        last = n
    end
    iv = firstindex(v) - 1

    if !iszero(rtol)
        # Determine the outcome on a scratch copy before touching `Q` or the stored factor:
        # `_NoQ` discards the rotations `_absorb_spike!` mirrors onto it, and only `RS` records
        # the candidate triangular factor. `zs` reuses `F.corr` and `RS` reuses `F.rscratch`:
        # nothing below reads `corr`'s contents again once `_project!` has returned, and both
        # are already sized to capacity.
        RS = view(F.rscratch, 1:(n + 1), 1:n)
        copyto!(RS, RA)
        zs = view(F.corr, 1:(n + 1))
        copyto!(zs, z)
        _absorb_spike!(RS, zs, _NoQ{T}(), v, iv, n, last)
        for j in 1:n
            colnorm = norm(view(RS, 1:j, j))
            abs(RS[j, j]) > rtol * colnorm && continue
            fill!(r, zero(T))
            throw(
                ArgumentError(
                    "column $j of the updated factorization is rank deficient: " *
                        "abs(R[$j,$j]) = $(abs(RS[j, j])) is at or below " *
                        "rtol * norm(column $j) = $(rtol * colnorm)"
                )
            )
        end
    end

    _absorb_spike!(RA, z, q, v, iv, n, last)
    # Re-establish zero storage outside the active block: the augmentation column of `q` and
    # row `n + 1` of `R` are working space this verb writes into, and nothing above assumes
    # they were already clean on entry.
    Rfull = getfield(F, :factors)
    fill!(view(Rfull, (n + 1):size(Rfull, 1), :), zero(T))
    fill!(view(Rfull, :, (n + 1):size(Rfull, 2)), zero(T))
    fill!(view(q.buf, :, (n + 1):size(q.buf, 2)), zero(T))
    return F
end

"""
    insert_column!(F::UpdatableQR, j, x; rtol = sqrt(eps(real(T)))) -> F

Insert `x` as column `j` of the factored matrix, in `O(mn + n|n + 1 - j|)` operations. `x` is
not modified.

The part of `x` orthogonal to the existing columns becomes the new column of `Q`, and its norm
`rho` becomes the new diagonal entry of `R`. `ArgumentError` is thrown when `rho` falls at or
below `rtol * norm(x)`: `rho` scales with `x`, so the threshold is relative, and the ratio it
tests lies in `[0, 1]`. A column that lies in the range of the existing ones leaves the
factorization exact but its last diagonal entry at the level of rounding noise, so a solve
through it divides by noise. `rtol = 0` admits every column whose residual is nonzero, which an
exactly dependent column is: measured, an exact copy of an existing column leaves `rho` at
1.95e-16 rather than at zero.

`DimensionMismatch` is thrown when the factorization is square, because the type admits only
`m >= n`.

Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating the
Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772-795.
"""
function insert_column!(
        F::UpdatableQR{T, S, <:DenseQ}, j::Integer, x::AbstractVector;
        rtol::Real = sqrt(eps(real(T)))
    ) where {T, S}
    m, n = F.m, F.n
    1 <= j <= n + 1 || throw(BoundsError(F, j))
    length(x) == m ||
        throw(DimensionMismatch("x has length $(length(x)), factorization is $(m)x$(n)"))
    m > n || throw(
        DimensionMismatch(
            "inserting a column would leave a $(m)x$(n + 1) factorization; " *
                "the factorization requires m >= n"
        )
    )
    q = getfield(F, :qrep)
    Qa = _active(q)
    r = _spare(q)
    w = view(F.work, 1:n)
    corr = view(F.corr, 1:n)
    ix = firstindex(x) - 1
    for i in 1:m
        r[i] = x[ix + i]
    end
    xnrm = norm(r)
    rho = _project!(w, r, Qa, corr)
    if !(rho > rtol * xnrm)
        # The projection built the candidate direction in the augmentation column. Clearing it
        # is what leaves the factorization as it was, and the next verb builds there.
        _clearspare!(q)
        throw(ArgumentError("column $j lies in the range of the existing columns: rho = $rho"))
    end
    r ./= rho
    # Growth comes after the guard, so a rejected insertion does not change the capacity.
    # `_grow!` carries the augmentation column into the new buffer, so the direction built
    # through `r` survives; the view itself does not, and nothing below reads it. `w` stays
    # valid because `_grow!` resizes `F.work` rather than rebinding it.
    _grow!(F, m, n + 1)
    R = getfield(F, :factors)
    for k in 1:n
        R[k, n + 1] = w[k]
    end
    R[n + 1, n + 1] = rho
    F.n = n + 1
    q.n = n + 1
    j != n + 1 && return shift_columns!(F, n + 1, Int(j))
    # `shift_columns!` re-establishes the zero invariant beyond the (new) active block on the
    # path above; appending at the end takes no such call, so it must do so itself here.
    fill!(view(R, (F.n + 1):size(R, 1), :), zero(T))
    fill!(view(R, :, (F.n + 1):size(R, 2)), zero(T))
    fill!(view(q.buf, :, (F.n + 1):size(q.buf, 2)), zero(T))
    return F
end

"""
    insert_row!(F::UpdatableQR, i, x) -> F

Insert `x` as row `i` of the factored matrix, in `O(mn + n^2)` operations. `x` is the new row,
not its adjoint, and is not modified.

A zero row opened in `Q` at position `i`, with the unit vector `e_i` as an extra column, extends
the factorization to the taller matrix; `n` rotations then return the appended row of `R` to
zero and the extra column of `Q` is dropped.

This is the verb that grows the row capacity, which re-strides every column of the stored
factor. A caller that inserts rows in a loop should pre-size with `capacity`.

Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5.
"""
function insert_row!(
        F::UpdatableQR{T, S, <:DenseQ}, i::Integer, x::AbstractVector
    ) where {T, S}
    m, n = F.m, F.n
    1 <= i <= m + 1 || throw(BoundsError(F, i))
    length(x) == n ||
        throw(DimensionMismatch("x has length $(length(x)), factorization is $(m)x$(n)"))
    # Grow before taking any view: growth rebinds both buffers.
    _grow!(F, m + 1, n)
    q = getfield(F, :qrep)
    _insertrow!(q, Int(i))
    R = getfield(F, :factors)
    RA = view(R, 1:(n + 1), 1:n)
    ix = firstindex(x) - 1
    for k in 1:n
        R[n + 1, k] = x[ix + k]
    end
    _spare(q)[i] = one(T)
    for k in 1:n
        c, s, rr = givensAlgorithm(R[k, k], R[n + 1, k])
        G = Givens(k, n + 1, oftype(R[k, k], c), oftype(R[k, k], s))
        lmul!(G, RA)
        rmul!(q, G')
        R[k, k] = rr
        R[n + 1, k] = zero(T)
    end
    F.m = m + 1
    # Re-establish zero storage outside the active block: `n` does not change here, so
    # nothing above assumes Q's trailing columns or R's spare row and column were already
    # zero.
    fill!(view(R, (n + 1):size(R, 1), :), zero(T))
    fill!(view(R, :, (n + 1):size(R, 2)), zero(T))
    fill!(view(q.buf, :, (n + 1):size(q.buf, 2)), zero(T))
    return F
end
