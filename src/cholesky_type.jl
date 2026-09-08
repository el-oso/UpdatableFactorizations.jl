"""
    UpdatableCholesky(C::Cholesky; capacity = 2size(C, 1))
    UpdatableCholesky(A::AbstractMatrix; uplo = :L, capacity = 2size(A, 1))

Cholesky factorization that supports rank-1 update and downdate and symmetric insertion,
deletion and shifting of indices. `capacity` is the largest size the factorization can reach
before its storage is reallocated.
"""
mutable struct UpdatableCholesky{T, R <: Real, S <: AbstractMatrix{T}} <: Factorization{T}
    factors::S
    n::Int
    uplo::Char
    work::Vector{T}      # the update vector, consumed in place
    cosines::Vector{R}   # downdate rotation cosines
    rot::Vector{T}       # downdate rotation sines
end

function UpdatableCholesky(C::Cholesky{T}; capacity::Int = 2size(C, 1)) where {T}
    n = size(C, 1)
    R = real(T)
    capacity >= n || throw(ArgumentError("capacity $capacity is below the size $n"))
    f = zeros(T, capacity, capacity)
    # Cholesky.factors only guarantees the stored triangle; LAPACK leaves the factored matrix in
    # the other one. Copying the stored triangle alone keeps the unstored half a true zero,
    # which the resizing kernels rely on.
    if C.uplo == 'L'
        for j in 1:n, i in j:n
            f[i, j] = C.factors[i, j]
        end
    else
        for j in 1:n, i in 1:j
            f[i, j] = C.factors[i, j]
        end
    end
    return UpdatableCholesky{T, R, Matrix{T}}(
        f, n, C.uplo, zeros(T, capacity), zeros(R, capacity), zeros(T, capacity)
    )
end

UpdatableCholesky(A::AbstractMatrix; uplo::Symbol = :L, capacity::Int = 2size(A, 1)) =
    UpdatableCholesky(cholesky(Hermitian(A, uplo)); capacity)

# The active factor presented as lower triangular. When the storage holds U with A = U'U, its
# adjoint is the lower factor L = U', and assignments through it conjugate as they must.
_lower(F::UpdatableCholesky) =
    F.uplo == 'L' ? view(F.factors, 1:F.n, 1:F.n) : adjoint(view(F.factors, 1:F.n, 1:F.n))

Base.size(F::UpdatableCholesky) = (F.n, F.n)
Base.size(F::UpdatableCholesky, i::Integer) = i <= 2 ? F.n : 1
Base.Matrix(F::UpdatableCholesky) = LowerTriangular(Matrix(_lower(F)))

capacity(F::UpdatableCholesky) = size(F.factors, 1)

function LinearAlgebra.ldiv!(F::UpdatableCholesky, b::AbstractVecOrMat)
    L = LowerTriangular(_lower(F))
    ldiv!(L, b)
    ldiv!(L', b)
    return b
end

Base.:\(F::UpdatableCholesky, b::AbstractVecOrMat) = ldiv!(F, copy(b))

LinearAlgebra.logdet(F::UpdatableCholesky) = 2 * sum(i -> log(real(_lower(F)[i, i])), 1:F.n)
LinearAlgebra.det(F::UpdatableCholesky) = exp(logdet(F))
