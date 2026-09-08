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
    for r in 1:n
        R[r, n] = zero(T)
    end
    n == 1 || _retriangularize!(view(R, 1:n, 1:(n - 1)), q)
    _dropcolumn!(q)
    F.n = n - 1
    return F
end

function insert_row! end
function delete_row! end
