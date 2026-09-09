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
