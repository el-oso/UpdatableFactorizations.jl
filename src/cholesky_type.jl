"""
    UpdatableCholesky(C::Cholesky; capacity = 2size(C, 1))
    UpdatableCholesky(A::AbstractMatrix; uplo = :L, capacity = 2size(A, 1))

Cholesky factorization that supports rank-1 update and downdate and symmetric insertion,
deletion and shifting of indices. `capacity` is the largest size the factorization can reach
before its storage is reallocated.

The lower factor `L` of `A = L*L'` is what is stored, whichever triangle the input holds.
`F.L` is that factor, `F.U` its adjoint, and `Matrix(F)` the reconstructed `A`.
"""
mutable struct UpdatableCholesky{T, R <: Real, S <: AbstractMatrix{T}} <: Factorization{T}
    factors::S
    n::Int
    work::Vector{T}      # scratch: the update vector, consumed in place
    cosines::Vector{R}   # scratch: downdate rotation cosines
    rot::Vector{T}       # scratch: downdate rotation sines, and the reordered insertion vector
    perm::Vector{Int}    # scratch: the index permutation, consumed in place
end

function UpdatableCholesky(C::Cholesky{T}; capacity::Int = 2size(C, 1)) where {T}
    n = size(C, 1)
    R = real(T)
    capacity >= n || throw(ArgumentError("capacity $capacity is below the size $n"))
    f = zeros(T, capacity, capacity)
    # Cholesky.factors only guarantees the stored triangle; LAPACK leaves the factored matrix in
    # the other one. Copying the stored triangle alone keeps the unstored half a true zero,
    # which the resizing kernels rely on. An upper factor U is transposed on the way in, so the
    # storage always holds the lower factor L = U'.
    if C.uplo == 'L'
        for j in 1:n, i in j:n
            f[i, j] = C.factors[i, j]
        end
    else
        for j in 1:n, i in j:n
            f[i, j] = conj(C.factors[j, i])
        end
    end
    return UpdatableCholesky{T, R, Matrix{T}}(
        f, n, zeros(T, capacity), zeros(R, capacity), zeros(T, capacity), zeros(Int, capacity)
    )
end

UpdatableCholesky(A::AbstractMatrix; uplo::Symbol = :L, capacity::Int = 2size(A, 1)) =
    UpdatableCholesky(cholesky(Hermitian(A, uplo)); capacity)

# The active block of the stored lower factor.
_lower(F::UpdatableCholesky) = view(F.factors, 1:F.n, 1:F.n)

Base.size(F::UpdatableCholesky) = (F.n, F.n)
function Base.size(F::UpdatableCholesky, dim::Integer)
    dim < 1 && throw(ArgumentError("dimension must be positive, got $dim"))
    return dim <= 2 ? F.n : 1
end

function Base.getproperty(F::UpdatableCholesky, s::Symbol)
    s === :L && return LowerTriangular(_lower(F))
    s === :U && return UpperTriangular(adjoint(_lower(F)))
    return getfield(F, s)
end

Base.propertynames(::UpdatableCholesky, private::Bool = false) =
    private ? (:L, :U, fieldnames(UpdatableCholesky)...) : (:L, :U)

Base.AbstractMatrix(F::UpdatableCholesky) = (L = F.L; L * L')
Base.Matrix(F::UpdatableCholesky) = Matrix(AbstractMatrix(F))

"""
    capacity(F::UpdatableCholesky) -> Int

The largest size `F` can reach before its storage is reallocated.
"""
capacity(F::UpdatableCholesky) = size(F.factors, 1)

# Every operation either completes or throws with the factorization left as it was.
LinearAlgebra.issuccess(::UpdatableCholesky) = true

function LinearAlgebra.ldiv!(F::UpdatableCholesky, b::AbstractVecOrMat)
    L = LowerTriangular(_lower(F))
    ldiv!(L, b)
    ldiv!(L', b)
    return b
end

LinearAlgebra.logdet(F::UpdatableCholesky) =
    2 * sum(i -> log(real(F.factors[i, i])), 1:F.n; init = zero(real(eltype(F.factors))))
LinearAlgebra.det(F::UpdatableCholesky) = exp(logdet(F))
