"""
    UpdatableLU(G::LU)
    UpdatableLU(A::AbstractMatrix; pivot = RowMaximum())

LU factorization that supports rank-1 update. Stored as `P*A = L*Diagonal(d)*U` with `L` unit
lower triangular and `U` unit upper triangular; `F.L` and `F.U` reassemble the factors in the
form `LinearAlgebra.lu` returns.
"""
mutable struct UpdatableLU{T, S <: AbstractMatrix{T}} <: Factorization{T}
    Lf::S
    d::Vector{T}
    Uf::S
    p::Vector{Int}
    pivoted::Bool
    work::Vector{T}
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
    return UpdatableLU{T, Matrix{T}}(Lf, d, Uf, collect(G.p), G.p != 1:n, zeros(T, 2n))
end

UpdatableLU(A::AbstractMatrix; pivot = RowMaximum()) = UpdatableLU(lu(A, pivot))

Base.size(F::UpdatableLU) = (length(F.d), length(F.d))
Base.size(F::UpdatableLU, i::Integer) = i <= 2 ? length(F.d) : 1

function Base.getproperty(F::UpdatableLU, s::Symbol)
    s === :L && return UnitLowerTriangular(getfield(F, :Lf))
    s === :U && return Diagonal(getfield(F, :d)) * UnitUpperTriangular(getfield(F, :Uf))
    return getfield(F, s)
end

Base.propertynames(::UpdatableLU) = (:L, :U, :p, :pivoted)

function LinearAlgebra.ldiv!(F::UpdatableLU, b::AbstractVector)
    permute!(b, F.p)
    ldiv!(UnitLowerTriangular(getfield(F, :Lf)), b)
    b ./= getfield(F, :d)
    ldiv!(UnitUpperTriangular(getfield(F, :Uf)), b)
    return b
end

Base.:\(F::UpdatableLU, b::AbstractVector) = ldiv!(F, copy(b))

LinearAlgebra.det(F::UpdatableLU) =
    prod(getfield(F, :d)) * (isodd(_permutation_parity(getfield(F, :p))) ? -1 : 1)

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
