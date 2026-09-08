"""
    UpdatableQR(G::Union{QR, QRCompactWY}; capacity = (2size(G, 1), 2size(G, 2)))
    UpdatableQR(A::AbstractMatrix; capacity = (2size(A, 1), 2size(A, 2)))

Thin QR factorization `A = Q*R` of an `m x n` matrix with `m >= n`, supporting rank-1 update,
insertion, deletion and shifting of columns, and insertion and deletion of rows.

`capacity` is `(mcap, ncap)`, the largest shape the factorization reaches before its storage is
reallocated; a verb that exceeds it doubles the dimension that was exceeded. The buffers hold
one row and column beyond `ncap`, because every verb works in a basis one column wider than the
factorization; `capacity = (m, n)` therefore pre-sizes exactly. Growing `ncap` appends columns
to a column-major buffer, while growing `mcap` re-strides every existing column, so row growth
is the expensive direction.

`F.Q` and `F.R` are views of live storage: a later verb changes what an earlier one returned,
and writing through them changes the factorization. `Matrix(F)` reconstructs `A`. The diagonal
of `R` carries whatever sign or phase the rotations leave, as `LinearAlgebra.qr` does; the
quantity that is kept small is `norm(F.Q'F.Q - I)`.

Construction throws `ArgumentError` when a diagonal entry of `R` is exactly zero, which a
column that is structurally zero produces. A column that is only numerically dependent on the
others leaves a small nonzero entry there and is accepted; [`insert_column!`](@ref)'s `rtol`
is the relative test.

Every operation either completes or throws with the factorization left as it was, so
`issuccess(F)` is always `true`.

A thin QR does not determine the sign of its determinant, so `det`, `logdet` and `logabsdet`
are not defined, matching `LinearAlgebra.qr`. For a square factorization,
`sum(log ∘ abs, diag(F.R))` is `log(abs(det(A)))`.
"""
mutable struct UpdatableQR{T, S <: AbstractMatrix{T}, Q <: AbstractQRep{T}} <: Factorization{T}
    qrep::Q          # active region is the leading m x n block; column n+1 is spare and zero
    factors::S       # (ncap+1) x (ncap+1); active region is the leading n x n block, and row
    #                  and column n+1 are spare and zero
    m::Int
    n::Int
    work::Vector{T}  # scratch: projection coefficients, consumed in place
    corr::Vector{T}  # scratch: the reorthogonalization correction, consumed in place
end

function UpdatableQR(
        G::Union{QR{T}, QRCompactWY{T}};
        capacity::Tuple{Integer, Integer} = (2size(G, 1), 2size(G, 2))
    ) where {T}
    m, n = size(G)
    m >= n || throw(DimensionMismatch("factorization is $(m)x$(n); it requires m >= n"))
    mcap, ncap = Int(capacity[1]), Int(capacity[2])
    mcap >= m || throw(ArgumentError("row capacity $mcap is below the size $m"))
    ncap >= n || throw(ArgumentError("column capacity $ncap is below the size $n"))
    qbuf = zeros(T, mcap, ncap + 1)
    qv = view(qbuf, 1:m, 1:n)
    for k in 1:n
        qv[k, k] = one(T)
    end
    # Applying the stored reflectors to the leading columns of an identity forms the thin factor
    # without copying reflector storage into the buffer, so everything outside the active block
    # stays a true zero, which the updating verbs rely on.
    lmul!(G.Q, qv)
    rbuf = zeros(T, ncap + 1, ncap + 1)
    src = getfield(G, :factors)
    for j in 1:n, i in 1:j
        rbuf[i, j] = src[i, j]
    end
    for k in 1:n
        iszero(rbuf[k, k]) &&
            throw(ArgumentError("column $k is rank deficient: R[$k,$k] is zero"))
    end
    return UpdatableQR{T, Matrix{T}, DenseQ{T, Matrix{T}}}(
        DenseQ{T, Matrix{T}}(qbuf, m, n), rbuf, m, n,
        zeros(T, ncap + 1), zeros(T, ncap + 1)
    )
end

UpdatableQR(A::AbstractMatrix; capacity = (2size(A, 1), 2size(A, 2))) =
    UpdatableQR(qr(A); capacity)

"""
    UpdatableQR(Q::AbstractMatrix, R::AbstractMatrix; capacity = (2size(Q, 1), 2size(Q, 2)))

Wrap a thin factorization that has already been computed. `Q` is `m x n` with orthonormal
columns and `R` is `n x n` upper triangular; neither is checked beyond its shape and a
zero diagonal entry, so the caller owns the orthonormality of `Q`.
"""
function UpdatableQR(
        Q::AbstractMatrix{T}, R::AbstractMatrix{T};
        capacity::Tuple{Integer, Integer} = (2size(Q, 1), 2size(Q, 2))
    ) where {T}
    Base.require_one_based_indexing(Q, R)
    m, n = size(Q)
    m >= n || throw(DimensionMismatch("Q is $(m) by $(n); it requires m >= n"))
    size(R, 1) == size(R, 2) ||
        throw(DimensionMismatch("R is $(size(R, 1)) by $(size(R, 2)); it must be square"))
    size(R, 1) == n || throw(
        DimensionMismatch("Q is $(m) by $(n) and R is $(size(R, 1)) by $(size(R, 2))")
    )
    mcap, ncap = Int(capacity[1]), Int(capacity[2])
    mcap >= m || throw(ArgumentError("row capacity $mcap is below the size $m"))
    ncap >= n || throw(ArgumentError("column capacity $ncap is below the size $n"))
    qbuf = zeros(T, mcap, ncap + 1)
    copyto!(view(qbuf, 1:m, 1:n), Q)
    rbuf = zeros(T, ncap + 1, ncap + 1)
    for j in 1:n, i in 1:j
        rbuf[i, j] = R[i, j]
    end
    for k in 1:n
        iszero(rbuf[k, k]) &&
            throw(ArgumentError("column $k is rank deficient: R[$k,$k] is zero"))
    end
    return UpdatableQR{T, Matrix{T}, DenseQ{T, Matrix{T}}}(
        DenseQ{T, Matrix{T}}(qbuf, m, n), rbuf, m, n,
        zeros(T, ncap + 1), zeros(T, ncap + 1)
    )
end

"""
    qr_householder(A; capacity = (2size(A, 1), 2size(A, 2))) -> UpdatableQR

Factor the `m x n` matrix `A` with `m >= n` by Householder reflections and return it as an
`UpdatableQR`. The factorization is `LinearAlgebra.qr`'s; this adds the updatable storage
around it.

This is the accuracy-preferring construction path. Orthogonality of `Q` is at machine precision
whatever the condition number of `A`, which no Gram-Schmidt variant gives.
"""
qr_householder(A::AbstractMatrix; capacity = (2size(A, 1), 2size(A, 2))) =
    UpdatableQR(qr(A); capacity)

# The active block of the stored triangular factor, that block plus the augmentation row every
# verb builds in, and the spare column one verb parks a moved column in. All three are one
# concrete SubArray type.
_upper(F::UpdatableQR) = view(getfield(F, :factors), 1:F.n, 1:F.n)
_raug(F::UpdatableQR) = view(getfield(F, :factors), 1:(F.n + 1), 1:F.n)
_rspare(F::UpdatableQR) = view(getfield(F, :factors), 1:F.n, F.n + 1)

capacity(F::UpdatableQR) = capacity(getfield(F, :qrep))

Base.size(F::UpdatableQR) = (F.m, F.n)
function Base.size(F::UpdatableQR, dim::Integer)
    dim < 1 && throw(ArgumentError("dimension must be positive, got $dim"))
    return dim <= 2 ? size(F)[dim] : 1
end

function Base.getproperty(F::UpdatableQR{T, S, <:DenseQ}, s::Symbol) where {T, S}
    s === :Q && return _active(getfield(F, :qrep))
    s === :R && return UpperTriangular(_upper(F))
    return getfield(F, s)
end

Base.propertynames(::UpdatableQR, private::Bool = false) =
    private ? (:Q, :R, fieldnames(UpdatableQR)...) : (:Q, :R)

# Reconstruction goes through F.R, not the raw block: a test that compares Matrix(F) against a
# target is then also a test that the stored block is triangular.
Base.AbstractMatrix(F::UpdatableQR) = F.Q * F.R
Base.Matrix(F::UpdatableQR) = Matrix(AbstractMatrix(F))

# Every operation either completes or throws with the factorization left as it was.
LinearAlgebra.issuccess(::UpdatableQR) = true

# Enlarge the storage to hold an `mneeded x nneeded` factorization, doubling only the dimension
# that was exceeded. Both buffers are rebound, so a view taken before a call to this dangles.
# The augmentation column is carried across with the active block, so a verb may build its new
# direction there and grow afterwards.
function _grow!(F::UpdatableQR{T, S, <:DenseQ}, mneeded::Int, nneeded::Int) where {T, S}
    mcap, ncap = capacity(F)
    (mneeded <= mcap && nneeded <= ncap) && return F
    newm = mneeded <= mcap ? mcap : max(mneeded, 2mcap)
    newn = nneeded <= ncap ? ncap : max(nneeded, 2ncap)
    q = getfield(F, :qrep)
    qbuf = zeros(T, newm, newn + 1)
    copyto!(view(qbuf, 1:F.m, 1:(F.n + 1)), view(q.buf, 1:F.m, 1:(F.n + 1)))
    q.buf = qbuf
    if newn != ncap
        rbuf = zeros(T, newn + 1, newn + 1)
        copyto!(view(rbuf, 1:F.n, 1:F.n), _upper(F))
        F.factors = rbuf
        resize!(F.work, newn + 1)
        resize!(F.corr, newn + 1)
    end
    return F
end

"""
    ldiv!(y, F::UpdatableQR, b) -> y

Overwrite `y` with the least-squares solution `R \\ (Q'b)`. `b` has length `size(F, 1)` and `y`
length `size(F, 2)`. Allocates nothing.

Both arguments are indexed from 1. The updating verbs accept offset vectors because they copy
their argument into the factorization's own storage; the solve applies `Q'` to `b` in place and
has nowhere to put an `m`-length copy.
"""
function LinearAlgebra.ldiv!(y::AbstractVector, F::UpdatableQR, b::AbstractVector)
    Base.require_one_based_indexing(y, b)
    length(b) == F.m ||
        throw(DimensionMismatch("b has length $(length(b)), factorization is $(F.m)x$(F.n)"))
    length(y) == F.n ||
        throw(DimensionMismatch("y has length $(length(y)), factorization is $(F.m)x$(F.n)"))
    mul!(y, F.Q', b)
    ldiv!(UpperTriangular(_upper(F)), y)
    return y
end

"""
    ldiv!(F::UpdatableQR, B) -> B

Overwrite the leading `size(F, 2)` rows of each column of `B` with its least-squares solution,
matching `ldiv!(::QRCompactWY, ::AbstractVecOrMat)`. The trailing rows are left as they were.

`B` is indexed from 1, as it is in the three-argument method.
"""
function LinearAlgebra.ldiv!(F::UpdatableQR, B::AbstractVecOrMat)
    Base.require_one_based_indexing(B)
    size(B, 1) == F.m ||
        throw(DimensionMismatch("B has $(size(B, 1)) rows, factorization is $(F.m)x$(F.n)"))
    y = view(F.work, 1:F.n)
    for c in axes(B, 2)
        col = view(B, :, c)
        mul!(y, F.Q', col)
        ldiv!(UpperTriangular(_upper(F)), y)
        for k in 1:F.n
            col[k] = y[k]
        end
    end
    return B
end

"""
    \\(F::UpdatableQR, B) -> X

Least-squares solution of `F.Q * F.R * X = B`: `X` has `size(F, 2)` rows, whatever the shape of
`B`, matching `\\(::QRCompactWY, ::AbstractVecOrMat)`. The element type of the solution is the
promotion of `eltype(F)` and `eltype(B)`, as `\\(::Factorization, ::AbstractVecOrMat)` promotes,
so an integer or narrower right-hand side is solved at the wider type rather than truncated.

Unlike the two `ldiv!` methods, this does not reuse `F`'s scratch storage, which is fixed at
`eltype(F)`: the promoted element type of a wider right-hand side, such as a complex one against
a real `F`, would not fit it.
"""
function Base.:\(F::UpdatableQR, B::AbstractVecOrMat)
    Base.require_one_based_indexing(B)
    size(B, 1) == F.m ||
        throw(DimensionMismatch("B has $(size(B, 1)) rows, factorization is $(F.m)x$(F.n)"))
    TFB = typeof(oneunit(eltype(F)) \ oneunit(eltype(B)))
    Y = _project(F, B, TFB)
    ldiv!(UpperTriangular(_upper(F)), Y)
    return Y
end

_project(F::UpdatableQR, b::AbstractVector, ::Type{TFB}) where {TFB} =
    mul!(similar(b, TFB, Base.OneTo(F.n)), F.Q', b)
_project(F::UpdatableQR, B::AbstractMatrix, ::Type{TFB}) where {TFB} =
    mul!(similar(B, TFB, Base.OneTo(F.n), axes(B, 2)), F.Q', B)

# LinearAlgebra reinterprets a real factorization applied to a complex right-hand side through
# `\(::Factorization{T}, ::VecOrMat{Complex{T}}) where T<:BlasReal`, equally specific to the
# method above (concrete on the factorization, abstract on the right-hand side, versus abstract
# on the factorization, concrete on the right-hand side); neither dominates, so the two are
# ambiguous unless something more specific than both is added. The method above already solves
# a complex right-hand side against a real factorization correctly on its own, so this one only
# needs to break the tie in its favor.
Base.:\(F::UpdatableQR{T}, B::VecOrMat{Complex{T}}) where {T <: LinearAlgebra.BlasReal} =
    invoke(\, Tuple{UpdatableQR, AbstractVecOrMat}, F, B)
