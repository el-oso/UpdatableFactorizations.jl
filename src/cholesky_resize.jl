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
