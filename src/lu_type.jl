abstract type AbstractUpdatableLU{T} <: Factorization{T} end

"""
    UpdatableLU(G::LU)
    UpdatableLU(A::AbstractMatrix; pivot = RowMaximum())

LU factorization that supports rank-1 update. Stored as `P*A = L*Diagonal(d)*U` with `L` unit
lower triangular and `U` unit upper triangular; `F.L` and `F.U` return fresh copies of the
factors in the form `LinearAlgebra.lu` gives them.

A failed update leaves the factorization invalid: `issuccess(F)` is then `false`, `F.info` is
the column at which the update failed, and solving with or taking the determinant of `F`
throws. An invalid factorization can only be rebuilt, not repaired.
"""
mutable struct UpdatableLU{T, S <: AbstractMatrix{T}} <: AbstractUpdatableLU{T}
    Lf::S
    d::Vector{T}
    Uf::S
    p::Vector{Int}
    work::Vector{T}
    info::Int
end

function UpdatableLU(G::LU{T}) where {T}
    n = size(G, 1)
    Lf = Matrix(UnitLowerTriangular(G.factors))
    U = Matrix(UpperTriangular(G.factors))
    d = T[U[k, k] for k in 1:n]
    Uf = Matrix{T}(I, n, n)
    for k in 1:n
        iszero(d[k]) && throw(ZeroPivotException(k))
        for j in (k + 1):n
            Uf[k, j] = U[k, j] / d[k]
        end
    end
    # `work` holds both of the rank-1 update's consumed vectors, so the update allocates nothing.
    return UpdatableLU{T, Matrix{T}}(Lf, d, Uf, collect(G.p), zeros(T, 2n), 0)
end

UpdatableLU(A::AbstractMatrix; pivot = RowMaximum()) = UpdatableLU(lu(A, pivot))

Base.size(F::UpdatableLU) = (length(getfield(F, :d)), length(getfield(F, :d)))
function Base.size(F::UpdatableLU, dim::Integer)
    dim < 1 && throw(ArgumentError("dimension must be positive, got $dim"))
    return dim <= 2 ? length(getfield(F, :d)) : 1
end

function Base.getproperty(F::UpdatableLU, s::Symbol)
    s === :L && return UnitLowerTriangular(copy(getfield(F, :Lf)))
    s === :U && return UpperTriangular(getfield(F, :d) .* getfield(F, :Uf))
    return getfield(F, s)
end

Base.propertynames(::UpdatableLU, private::Bool = false) =
    private ? (:L, :U, fieldnames(UpdatableLU)...) : (:L, :U, :p, :info)

LinearAlgebra.issuccess(F::UpdatableLU) = iszero(getfield(F, :info))

function _checkvalid(F::UpdatableLU)
    info = getfield(F, :info)
    iszero(info) || throw(
        ArgumentError(
            "factorization is invalid: an update failed at column $info, rebuild it"
        )
    )
    return nothing
end

function LinearAlgebra.ldiv!(F::UpdatableLU, B::AbstractVecOrMat)
    _checkvalid(F)
    p = getfield(F, :p)
    for c in axes(B, 2)
        permute!(view(B, :, c), p)
    end
    ldiv!(UnitLowerTriangular(getfield(F, :Lf)), B)
    B ./= getfield(F, :d)
    ldiv!(UnitUpperTriangular(getfield(F, :Uf)), B)
    return B
end

Base.AbstractMatrix(F::UpdatableLU) = (F.L * F.U)[invperm(getfield(F, :p)), :]
Base.Matrix(F::UpdatableLU) = Matrix(AbstractMatrix(F))

function LinearAlgebra.det(F::UpdatableLU{T}) where {T}
    _checkvalid(F)
    v = prod(getfield(F, :d); init = one(T))
    return isodd(_permutation_parity(getfield(F, :p))) ? -v : v
end

function LinearAlgebra.logabsdet(F::UpdatableLU{T}) where {T}
    _checkvalid(F)
    m = zero(real(T))
    s = isodd(_permutation_parity(getfield(F, :p))) ? -one(T) : one(T)
    for x in getfield(F, :d)
        m += log(abs(x))
        s *= x / abs(x)
    end
    return m, s
end

LinearAlgebra.logdet(F::UpdatableLU) = ((m, s) = logabsdet(F); m + log(s))

function _permutation_parity(p::AbstractVector{Int})
    q = collect(p)
    swaps = 0
    for i in eachindex(q)
        while q[i] != i
            j = q[i]
            q[i], q[j] = q[j], q[i]
            swaps += 1
        end
    end
    return swaps
end
