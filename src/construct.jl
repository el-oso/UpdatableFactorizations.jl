"""
    default_rankk!(C, A, alpha, beta; uplo = 'L') -> C

Overwrite `C` with `alpha*A*A' + beta*C`. On BLAS element types only the triangle named by
`uplo` is read and written, through `syrk!` for real `A` and `herk!` for complex `A`. For any
other element type both triangles are written, because there is no symmetric rank-k kernel to
call and a full product costs less than assembling one triangle entry by entry.

This is the default deferred flush of `cholesky_crout`.
"""
function default_rankk!(
        C::StridedMatrix{T}, A::StridedMatrix{T}, alpha, beta; uplo::Char = 'L'
    ) where {T <: LinearAlgebra.BlasReal}
    return BLAS.syrk!(uplo, 'N', T(alpha), A, T(beta), C)
end

function default_rankk!(
        C::StridedMatrix{T}, A::StridedMatrix{T}, alpha, beta; uplo::Char = 'L'
    ) where {T <: LinearAlgebra.BlasComplex}
    # herk! takes real scalars; converting straight to R (rather than through `real(...)`) relies
    # on Julia's Complex-to-Real conversion to throw InexactError on a nonzero imaginary part
    # instead of silently discarding it.
    R = real(T)
    return BLAS.herk!(uplo, 'N', R(alpha), A, R(beta), C)
end

default_rankk!(C, A, alpha, beta; uplo::Char = 'L') = mul!(C, A, A', alpha, beta)

"""
    cholesky_crout(A; s = 64, uplo = :L, capacity = 2size(A, 1), rankk! = default_rankk!)

Factor the Hermitian positive definite matrix `A` as `L*L'` and return an `UpdatableCholesky`.
Only the triangle named by `uplo` is read; a `Symmetric` or `Hermitian` argument must be one
that stores that triangle. Columns are formed one at a time and the trailing update is deferred,
then flushed every `s` columns through `rankk!(C, A, alpha, beta; uplo)`, which defaults to a
symmetric rank-k update. Throws `PosDefException(c)` at the first column whose pivot is not
positive.

This is slower than `LinearAlgebra.cholesky`, which calls LAPACK's blocked `potrf`; see the
construction page of the documentation for the measured ratios.

Camarero, *Simple, Fast and Practicable Algorithms for Cholesky, LU and QR Decomposition Using
Fast Rectangular Matrix Multiplication*, arXiv:1812.02056 (2018), Algorithm 1.
"""
function cholesky_crout(
        A::AbstractMatrix{T}; s::Int = 64, uplo::Symbol = :L,
        capacity::Int = 2size(A, 1), rankk! = default_rankk!
    ) where {T}
    Base.require_one_based_indexing(A)
    n = LinearAlgebra.checksquare(A)
    s >= 1 || throw(ArgumentError("block size s must be at least 1, got $s"))
    M = Matrix(Hermitian(A, uplo))
    L = zeros(T, n, n)
    panel = Vector{T}(undef, n)
    z = 1
    @views for c in 1:n
        if c == z + s
            rankk!(M[c:n, c:n], L[c:n, z:(c - 1)], -one(T), one(T); uplo = 'L')
            z = c
        end
        # The conjugated row panel: column c subtracts L[i, k] * conj(L[c, k]) over the block.
        p = panel[1:(c - z)]
        for k in eachindex(p)
            p[k] = conj(L[c, z + k - 1])
        end
        acc = real(M[c, c]) - sum(abs2, p; init = zero(real(T)))
        acc > 0 || throw(PosDefException(c))
        Lcc = sqrt(acc)
        L[c, c] = Lcc
        if c < n
            col = L[(c + 1):n, c]
            copyto!(col, M[(c + 1):n, c])
            c > z && mul!(col, L[(c + 1):n, z:(c - 1)], p, -one(T), one(T))
            col ./= Lcc
        end
    end
    return UpdatableCholesky(Cholesky(L, 'L', 0); capacity)
end

# Index within `col` of the row to move into the pivot position.
_pivotrow(::NoPivot, col) = 1
_pivotrow(::RowMaximum, col) = findmax(abs, col)[2]
_pivotrow(pivot, col) = throw(
    ArgumentError("pivoting strategy $pivot is not supported; use NoPivot() or RowMaximum()")
)

"""
    lu_crout(A; s = 64, pivot = RowMaximum(), rtol = 0, matmul! = mul!)

Factor the square matrix `A` as `P*A = L*U` and return an `UpdatableLU`. Columns of `L` and rows
of `U` are formed one at a time and the trailing update is deferred, then flushed every `s`
columns through `matmul!(C, A, B, alpha, beta)`, which defaults to `LinearAlgebra.mul!`.

`pivot` is `RowMaximum()` for partial pivoting or `NoPivot()` for the unpivoted factorization,
which exists only when every leading principal minor is nonzero. Either way a pivot of
magnitude at most `rtol` times the norm of its column raises `ZeroPivotException(c)` naming the
column; the default `rtol = 0` makes that an exact-zero test.

The pivoted form, which is the default, is slower than `LinearAlgebra.lu`, which calls LAPACK's
blocked `getrf`. The unpivoted form has no blocked counterpart in the standard library:
`lu!(A, NoPivot())` is an unblocked generic fallback. See the construction page of the
documentation for the measured ratios.

Camarero, *Simple, Fast and Practicable Algorithms for Cholesky, LU and QR Decomposition Using
Fast Rectangular Matrix Multiplication*, arXiv:1812.02056 (2018), Algorithm 2.
"""
function lu_crout(
        A::AbstractMatrix{T}; s::Int = 64, pivot = RowMaximum(),
        rtol::Real = 0, matmul! = mul!
    ) where {T}
    Base.require_one_based_indexing(A)
    n = LinearAlgebra.checksquare(A)
    s >= 1 || throw(ArgumentError("block size s must be at least 1, got $s"))
    M = Matrix{T}(undef, n, n)
    copyto!(M, A)
    L = zeros(T, n, n)
    U = Matrix{T}(I, n, n)
    ipiv = collect(1:n)
    z = 1
    @views for c in 1:n
        if c == z + s
            matmul!(M[c:n, c:n], L[c:n, z:(c - 1)], U[z:(c - 1), c:n], -one(T), one(T))
            z = c
        end
        col = L[c:n, c]
        copyto!(col, M[c:n, c])
        c > z && matmul!(col, L[c:n, z:(c - 1)], U[z:(c - 1), c], -one(T), one(T))
        pv = c + _pivotrow(pivot, col) - 1
        if pv != c
            # The swaps need an explicit temporary: `@views` rewrites an indexed left-hand side
            # in a tuple assignment to `Base.maybeview(...)`, which the parser then rejects as
            # a function definition.
            for j in 1:c
                t = L[c, j]
                L[c, j] = L[pv, j]
                L[pv, j] = t
            end
            for j in (c + 1):n
                t = M[c, j]
                M[c, j] = M[pv, j]
                M[pv, j] = t
            end
            ipiv[c] = pv
        end
        Lcc = L[c, c]
        (iszero(Lcc) || abs(Lcc) <= rtol * norm(col)) && throw(ZeroPivotException(c))
        if c < n
            row = U[c, (c + 1):n]
            copyto!(row, M[c, (c + 1):n])
            c > z &&
                matmul!(row, transpose(U[z:(c - 1), (c + 1):n]), L[c, z:(c - 1)], -one(T), one(T))
            row ./= Lcc
        end
    end
    # The standard library packs L and U into one array with a unit-diagonal L, which is the
    # form UpdatableLU splits apart again.
    f = Matrix{T}(undef, n, n)
    for j in 1:n
        for i in 1:(j - 1)
            f[i, j] = L[i, i] * U[i, j]
        end
        f[j, j] = L[j, j]
        for i in (j + 1):n
            f[i, j] = L[i, j] / L[j, j]
        end
    end
    return UpdatableLU(LU{T}(f, ipiv, 0))
end

"""
    qr_bcgs(A; s = 64, reorth = true, rtol = 0, capacity = (2m, 2n), matmul! = mul!) -> UpdatableQR

Factor the `m` by `n` matrix `A` with `m >= n` as `Q*R` by block classical Gram-Schmidt and
return it as an [`UpdatableQR`](@ref) holding the thin `Q` (`m` by `n`, orthonormal columns) and
the upper triangular `R`, ready for the updating verbs. Columns are orthogonalized one at a time
against the current block and the accumulated projection against earlier blocks is deferred,
then flushed every `s` columns through `matmul!(C, A, B, alpha, beta)`, which defaults to
`LinearAlgebra.mul!`. `capacity` is passed through to the returned factorization.

`reorth = true` runs the projection a second time at every flush and at every column, which
costs roughly twice as much and gives orthogonality of order `eps` provided the product of the
condition number and `eps` is well below one. `reorth = false` leaves the loss of orthogonality
growing as the square of the condition number. [`qr_householder`](@ref) is more accurate than
either and is the recommended path; see the construction page of the documentation for the
measured ratios and residuals.

A column whose projection against the columns before it falls to `rtol` times its own norm
raises `ArgumentError` naming the column, which means the matrix is rank deficient and a
Gram-Schmidt factorization of it does not exist. The default `rtol = 0` makes that an exact
collapse; a rank-deficient input in floating point leaves rounding noise instead, so detecting
it needs an `rtol` above the noise, of order the condition number times `eps`.

Camarero, *Simple, Fast and Practicable Algorithms for Cholesky, LU and QR Decomposition Using
Fast Rectangular Matrix Multiplication*, arXiv:1812.02056 (2018), Algorithm 3. The
reorthogonalization bound is Giraud, Langou and Rozložník, *The loss of orthogonality in the
Gram-Schmidt orthogonalization process*, Computers and Mathematics with Applications 50 (2005),
1069-1075.
"""
function qr_bcgs(
        A::AbstractMatrix{T}; s::Int = 64, reorth::Bool = true, rtol::Real = 0,
        capacity::Tuple{Integer, Integer} = (2size(A, 1), 2size(A, 2)), matmul! = mul!
    ) where {T}
    Base.require_one_based_indexing(A)
    m, n = size(A)
    m >= n || throw(DimensionMismatch("A is $m by $n; qr_bcgs requires m >= n"))
    s >= 1 || throw(ArgumentError("block size s must be at least 1, got $s"))
    M = Matrix{T}(undef, m, n)
    copyto!(M, A)
    Q = zeros(T, m, n)
    coef = Matrix{T}(undef, min(s, n), n)
    z = 1
    @views for c in 1:n
        if c == z + s
            BQ = Q[:, z:(c - 1)]
            BM = M[:, c:n]
            C = coef[1:s, 1:(n - c + 1)]
            matmul!(C, BQ', BM, one(T), zero(T))
            matmul!(BM, BQ, C, -one(T), one(T))
            if reorth
                matmul!(C, BQ', BM, one(T), zero(T))
                matmul!(BM, BQ, C, -one(T), one(T))
            end
            z = c
        end
        v = M[:, c]
        for _ in 1:(reorth ? 2 : 1)
            for j in z:(c - 1)
                u = Q[:, j]
                axpy!(-dot(u, v), u, v)
            end
        end
        nv = norm(v)
        # The threshold is relative to the original column, not to `v`: by this point the
        # deferred flush has already projected `v` against every earlier block.
        (iszero(nv) || nv <= rtol * norm(A[:, c])) &&
            throw(ArgumentError("column $c is a combination of the columns before it"))
        v ./= nv
        copyto!(Q[:, c], v)
    end
    R = Matrix{T}(undef, n, n)
    matmul!(R, Q', A, one(T), zero(T))
    for j in 1:n, i in (j + 1):n
        R[i, j] = zero(T)
    end
    return UpdatableQR(Q, R; capacity)
end
