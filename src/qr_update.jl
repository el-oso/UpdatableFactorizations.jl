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

function insert_row! end
function delete_row! end

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
noise, so a later solve through it divides by that noise. `rtol = 0` skips the check entirely,
admitting any update including an exactly singular one, matching a plain re-triangularization
with no rank check and costing nothing beyond it.

When `rtol` is nonzero, the candidate `R` is computed on a scratch copy, and the check runs
against it, before `Q` or the stored factor are touched; a thrown update therefore leaves the
factorization exactly as it was.

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
        # `dummy` is a zero-row `DenseQ` sharing `RA`'s element type, so the rotations
        # `_absorb_spike!` mirrors onto it are no-ops, and only `RS` records the candidate
        # triangular factor. `zs` reuses `F.corr`: nothing below reads `corr`'s contents again
        # once `_project!` has returned, and it is already sized to `ncap + 1`.
        RS = copy(RA)
        zs = view(F.corr, 1:(n + 1))
        copyto!(zs, z)
        dummybuf = similar(q.buf, 0, n + 1)
        dummy = DenseQ{T, typeof(dummybuf)}(dummybuf, 0, n)
        _absorb_spike!(RS, zs, dummy, v, iv, n, last)
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
