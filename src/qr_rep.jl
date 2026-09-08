"""
    AbstractQRep{T}

Orthonormal factor of an [`UpdatableQR`](@ref), `m x n` with `m >= n`. A representation stores
whatever it likes, so long as it answers the five operations of its contract; `Base.Matrix` and
`Base.size(q, dim)` are derived from those and need no per-representation method.

A thin factor does not subtype `LinearAlgebra.AbstractQ`, whose interface is the implicit square
factor: `size(qr(randn(6, 3)).Q)` is `(6, 6)`.
"""
abstract type AbstractQRep{T} end

"""
    materialize(q::AbstractQRep) -> AbstractQRep

An explicitly stored representation equal to `q`. A representation that is already explicit
returns itself.
"""
function materialize end

@contract AbstractQRep{T} "The orthonormal factor of an UpdatableQR." begin
    Base.size(::Self)::Tuple{Int,Int} => "the shape of the active factor"
    Base.copyto!(::AbstractMatrix, ::Self) => "write the active factor densely"
    materialize(::Self) => "an explicitly stored equivalent"
    LinearAlgebra.lmul!(::Givens, ::Self) => "apply a rotation on the left"
    LinearAlgebra.rmul!(::Self, ::Givens) => "apply a rotation on the right"
end

Base.eltype(::AbstractQRep{T}) where {T} = T
Base.Matrix(q::AbstractQRep{T}) where {T} = copyto!(Matrix{T}(undef, size(q)), q)

function Base.size(q::AbstractQRep, dim::Integer)
    dim < 1 && throw(ArgumentError("dimension must be positive, got $dim"))
    return dim <= 2 ? size(q)[dim] : 1
end

"""
    DenseQ(buf, m, n)

Orthonormal factor held explicitly: the leading `m x n` block of `buf`, with `m >= n`. Storage
outside that block is zero. Column `n+1` of `buf` always exists — it is the augmentation the
updating verbs build in — and is zero except while a verb runs.
"""
mutable struct DenseQ{T,S<:AbstractMatrix{T}} <: AbstractQRep{T}
    buf::S
    m::Int
    n::Int
end

# The active factor, the factor plus its augmentation column, and that column alone. All three
# are one concrete SubArray type, so every kernel has a single specialization.
_active(q::DenseQ) = view(q.buf, 1:q.m, 1:q.n)
_augmented(q::DenseQ) = view(q.buf, 1:q.m, 1:(q.n+1))
_spare(q::DenseQ) = view(q.buf, 1:q.m, q.n + 1)

Base.size(q::DenseQ) = (q.m, q.n)
Base.copyto!(A::AbstractMatrix, q::DenseQ) = copyto!(A, _active(q))
materialize(q::DenseQ) = q

# The largest shape reachable before the buffer is reallocated. The trailing column is the
# augmentation every verb works in, not capacity a caller may fill.
capacity(q::DenseQ) = (size(q.buf, 1), size(q.buf, 2) - 1)

LinearAlgebra.lmul!(G::Givens, q::DenseQ) = (lmul!(G, _augmented(q)); q)
LinearAlgebra.rmul!(q::DenseQ, G::Givens) = (rmul!(_augmented(q), G); q)

# Structural changes. Each re-establishes zero storage outside the active block itself,
# rather than assuming the caller already left it that way.
function _insertrow!(q::DenseQ{T}, i::Integer) where {T}
    for j in 1:q.n, r in q.m:-1:i
        q.buf[r+1, j] = q.buf[r, j]
    end
    for j in 1:q.n
        q.buf[i, j] = zero(T)
    end
    q.m += 1
    fill!(view(q.buf, :, q.n + 1), zero(T))
    return q
end

function _deleterow!(q::DenseQ{T}, i::Integer) where {T}
    for j in 1:q.n, r in i:(q.m-1)
        q.buf[r, j] = q.buf[r+1, j]
    end
    for j in 1:(q.n+1)
        q.buf[q.m, j] = zero(T)
    end
    for r in 1:q.m
        q.buf[r, q.n+1] = zero(T)
    end
    q.m -= 1
    return q
end

function _dropcolumn!(q::DenseQ{T}) where {T}
    q.n -= 1
    fill!(view(q.buf, :, (q.n+1):size(q.buf, 2)), zero(T))
    return q
end

_clearspare!(q::DenseQ{T}) where {T} = (fill!(_spare(q), zero(T)); q)

@verify DenseQ
