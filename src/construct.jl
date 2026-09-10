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

# Zero the region of `buf` outside its leading `nrows x ncols` block, so a target reused at a
# smaller size does not retain nonzero entries left over from a larger factorization it held
# before.
function _clear_outside!(buf::AbstractMatrix{T}, nrows::Int, ncols::Int) where {T}
    fill!(view(buf, (nrows + 1):size(buf, 1), :), zero(T))
    fill!(view(buf, :, (ncols + 1):size(buf, 2)), zero(T))
    return buf
end

# Core Crout Cholesky loop shared by `cholesky_crout` and `cholesky_crout!`. `L` is the n x n
# destination, with only entries where row >= column written; `M` is the working matrix,
# consumed and overwritten with the trailing Schur complement at each flush; `panel` is scratch
# of length at least n.
function _cholesky_crout_kernel!(L, M, n::Int, s::Int, rankk!, panel::AbstractVector{T}) where {T}
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
    return nothing
end

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
    capacity >= n || throw(ArgumentError("capacity $capacity is below the size $n"))
    Hermitian(A, uplo)   # throws when A is a Symmetric/Hermitian wrapper for the other triangle
    # Only the lower triangle (row >= col) is ever read below, whatever `uplo` names, so only
    # that triangle is populated; the upper triangle stays at its zero-initialized value and is
    # never touched again except by a `rankk!` that writes a full block, which reads back only
    # what it wrote and never feeds the factor itself.
    M = zeros(T, n, n)
    if uplo === :L
        for j in 1:n, i in j:n
            M[i, j] = A[i, j]
        end
    else
        for j in 1:n, i in j:n
            M[i, j] = conj(A[j, i])
        end
    end
    f = zeros(T, capacity, capacity)
    L = view(f, 1:n, 1:n)
    panel = Vector{T}(undef, n)
    _cholesky_crout_kernel!(L, M, n, s, rankk!, panel)
    return _wrap_cholesky(f, n)
end

"""
    cholesky_crout!(F::UpdatableCholesky, A; s = 64, uplo = :L, rankk! = default_rankk!) -> F

Factor the Hermitian positive definite matrix `A` as `L*L'` into the existing `F`, overwriting
its own storage. `A` is destroyed: it is used directly as the algorithm's working matrix,
exactly as `LinearAlgebra.cholesky!` destroys its argument. Throws `ArgumentError` naming
`capacity(F)` and `size(A, 1)` when `F`'s capacity cannot hold the result, rather than growing
it.

A caller who refactorizes repeatedly into the same `F`, at a size no larger than `capacity(F)`,
allocates nothing. See [`cholesky_crout`](@ref) for `s`, `uplo`, `rankk!` and the thrown
`PosDefException`.

On `PosDefException`, `F`'s storage has already been partially overwritten; unlike the updating
verbs, `issuccess(F)` does not reflect this, so `F` must be rebuilt rather than reused.
"""
function cholesky_crout!(
        F::UpdatableCholesky{T}, A::AbstractMatrix{T}; s::Int = 64, uplo::Symbol = :L,
        rankk! = default_rankk!
    ) where {T}
    Base.require_one_based_indexing(A)
    n = LinearAlgebra.checksquare(A)
    # `lazy"..."` rather than plain interpolation: two interpolated ArgumentErrors in one method
    # despecialize inference enough to box the arguments of the branch that is never taken,
    # which measurably allocates even though the throw itself does not execute.
    s >= 1 || throw(ArgumentError(lazy"block size s must be at least 1, got $s"))
    cap = capacity(F)
    cap >= n || throw(ArgumentError(lazy"capacity $cap is below the size $n"))
    Hermitian(A, uplo)   # throws when A is a Symmetric/Hermitian wrapper for the other triangle
    # `A` is consumed directly as the working matrix: for uplo = :L its own storage already is
    # the lower-triangle scratch the algorithm reads and updates; for uplo = :U the lazy
    # conjugate transpose gives the same view without copying.
    M = uplo === :L ? A : A'
    _clear_outside!(F.factors, n, n)
    L = view(F.factors, 1:n, 1:n)
    panel = view(F.work, 1:n)
    _cholesky_crout_kernel!(L, M, n, s, rankk!, panel)
    F.n = n
    return F
end

# Index within `col` of the row to move into the pivot position.
_pivotrow(::NoPivot, col) = 1
_pivotrow(::RowMaximum, col) = findmax(abs, col)[2]

# `iamax` is LAPACK's own pivot search and runs about thirty times faster than the generic scan.
# It is used only for real element types, where it maximizes the same quantity: for complex ones
# it maximizes |real| + |imag| rather than the modulus, which would pick a different pivot than
# `LinearAlgebra.generic_lufact!` does.
_pivotrow(::RowMaximum, col::StridedVector{<:LinearAlgebra.BlasReal}) = BLAS.iamax(col)
_pivotrow(pivot, col) = throw(
    ArgumentError("pivoting strategy $pivot is not supported; use NoPivot() or RowMaximum()")
)

# Core Crout LU loop shared by `lu_crout` and `lu_crout!`. `M` is the n x n working matrix,
# consumed and overwritten with the trailing Schur complement at each flush; `L` is the n x n
# destination with only entries where row >= column written; `Ut` is the n x n destination for
# the transpose of the upper triangular factor (`Ut[j, k] == U-factor[k, j]`), with only entries
# where row >= column written -- the same convention as `L`, so both fill contiguously down a
# column; `p` is filled with the pivot permutation, maintained directly by applying each row
# interchange to it as the interchange happens, rather than recorded as a separate encoding and
# composed into a permutation afterward.
function _lu_crout_kernel!(
        M::AbstractMatrix{T}, L, Ut, p::AbstractVector{Int}, n::Int, s::Int,
        pivot, rtol::Real, matmul!
    ) where {T}
    for i in 1:n
        p[i] = i
    end
    z = 1
    @views for c in 1:n
        if c == z + s
            matmul!(
                M[c:n, c:n], L[c:n, z:(c - 1)], transpose(Ut[c:n, z:(c - 1)]), -one(T), one(T)
            )
            z = c
        end
        col = L[c:n, c]
        copyto!(col, M[c:n, c])
        c > z && matmul!(col, L[c:n, z:(c - 1)], Ut[c, z:(c - 1)], -one(T), one(T))
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
            tp = p[c]
            p[c] = p[pv]
            p[pv] = tp
        end
        Lcc = L[c, c]
        (iszero(Lcc) || abs(Lcc) <= rtol * norm(col)) && throw(ZeroPivotException(c))
        if c < n
            col2 = Ut[(c + 1):n, c]
            copyto!(col2, M[c, (c + 1):n])
            c > z && matmul!(col2, Ut[(c + 1):n, z:(c - 1)], L[c, z:(c - 1)], -one(T), one(T))
            col2 ./= Lcc
        end
    end
    return nothing
end

# Divide the raw Crout factor's diagonal out into `d` and set it to one, turning `L` into the
# unit lower triangular factor `UpdatableLU` stores. `Ut` needs no equivalent step: it is already
# unit lower triangular, since its diagonal starts at one and the loop above only ever writes
# its strictly lower entries.
function _lu_finish!(L, d::AbstractVector{T}, n::Int) where {T}
    for j in 1:n
        d[j] = L[j, j]
        L[j, j] = one(T)
        for i in (j + 1):n
            L[i, j] /= d[j]
        end
    end
    return nothing
end

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
    Ut = Matrix{T}(I, n, n)
    p = Vector{Int}(undef, n)
    _lu_crout_kernel!(M, L, Ut, p, n, s, pivot, rtol, matmul!)
    d = Vector{T}(undef, n)
    _lu_finish!(L, d, n)
    # `work` holds both of the rank-1 update's consumed vectors, so the update allocates nothing.
    return UpdatableLU{T, Matrix{T}}(L, d, Ut, p, zeros(T, 2n), 0)
end

"""
    lu_crout!(F::UpdatableLU, A; s = 64, pivot = RowMaximum(), rtol = 0, matmul! = mul!) -> F

Factor the square matrix `A` as `P*A = L*U` into the existing `F`, overwriting its own storage.
`A` is destroyed: it is used directly as the algorithm's working matrix, exactly as
`LinearAlgebra.lu!` destroys its argument.

`UpdatableLU` has no capacity beyond its own size, so this throws `ArgumentError` naming
`size(F, 1)` and `size(A, 1)` when they differ, rather than resizing.

A caller who refactorizes repeatedly into the same `F`, at the same size every time, allocates
nothing. See [`lu_crout`](@ref) for `s`, `pivot`, `rtol`, `matmul!` and the thrown
`ZeroPivotException`.

On `ZeroPivotException`, `F`'s storage has already been partially overwritten; `issuccess(F)` is
`false` afterwards, matching the updating verbs, and `F` must be rebuilt rather than reused.
"""
function lu_crout!(
        F::UpdatableLU{T}, A::AbstractMatrix{T}; s::Int = 64, pivot = RowMaximum(),
        rtol::Real = 0, matmul! = mul!
    ) where {T}
    Base.require_one_based_indexing(A)
    n = LinearAlgebra.checksquare(A)
    # lazy"..." here for the same reason as in cholesky_crout!: plain interpolation in two
    # ArgumentErrors in one method measurably allocates even on the path that never throws.
    s >= 1 || throw(ArgumentError(lazy"block size s must be at least 1, got $s"))
    ntarget = size(F, 1)
    ntarget == n || throw(ArgumentError(lazy"capacity $ntarget is below the size $n"))
    L = getfield(F, :Lf)
    Ut = getfield(F, :Ut)
    d = getfield(F, :d)
    p = getfield(F, :p)
    try
        _lu_crout_kernel!(A, L, Ut, p, n, s, pivot, rtol, matmul!)
    catch e
        e isa ZeroPivotException && setfield!(F, :info, e.info)
        rethrow()
    end
    _lu_finish!(L, d, n)
    setfield!(F, :info, 0)
    return F
end

# Core block classical Gram-Schmidt loop shared by `qr_bcgs` and `qr_bcgs!`. `Q` and `R` are the
# m x n and n x n destinations; `M` is the working matrix, consumed and overwritten with each
# column's residual; `coef` is scratch of shape at least (min(s, n), n). `colnorms` holds each
# column's norm in the original matrix, captured before `M` is touched: a deferred flush can
# reduce a later column's entry in `M` before that column is reached, so when `M` aliases the
# caller's own matrix (the in-place path) the original norm would otherwise already be
# overwritten by the time the rank-deficiency test needs it. Real norms are boxed in `T` and
# unboxed with `real` where they are compared, so the same buffer serves whether `T` is real or
# complex.
function _qr_bcgs_kernel!(
        Q, R, M::AbstractMatrix{T}, colnorms::AbstractVector{T}, n::Int, s::Int,
        reorth::Bool, rtol::Real, matmul!, coef
    ) where {T}
    z = 1
    # Every coefficient R needs is computed somewhere in this loop: a flush's `C` is `BQ'*BM`
    # with `BQ` already orthogonal to every earlier block, so it equals the R block pairing the
    # finished block against the not-yet-processed columns; the intra-block `dot(u, v)` is the R
    # entry pairing two columns of the same block; `nv` is the diagonal. Summing them here, rather
    # than recomputing `Q'*A` once Q is complete, reconstructs each column of A as a telescoping
    # sum of exactly the pieces subtracted from it during elimination -- an identity that holds
    # regardless of how orthogonal the finished Q turns out to be, unlike a fresh `Q'*A`, which
    # amplifies error by the same loss of orthogonality it is trying to correct for.
    @views for c in 1:n
        if c == z + s
            BQ = Q[:, z:(c - 1)]
            BM = M[:, c:n]
            C = coef[1:s, 1:(n - c + 1)]
            Rblock = R[z:(c - 1), c:n]
            matmul!(C, BQ', BM, one(T), zero(T))
            copyto!(Rblock, C)
            matmul!(BM, BQ, C, -one(T), one(T))
            if reorth
                matmul!(C, BQ', BM, one(T), zero(T))
                Rblock .+= C
                matmul!(BM, BQ, C, -one(T), one(T))
            end
            z = c
        end
        v = M[:, c]
        for _ in 1:(reorth ? 2 : 1)
            for j in z:(c - 1)
                u = Q[:, j]
                coeff = dot(u, v)
                R[j, c] += coeff
                axpy!(-coeff, u, v)
            end
        end
        nv = norm(v)
        # The threshold is relative to the original column, not to `v`: by this point the
        # deferred flush has already projected `v` against every earlier block.
        (iszero(nv) || nv <= rtol * real(colnorms[c])) &&
            throw(ArgumentError("column $c is a combination of the columns before it"))
        R[c, c] = nv
        v ./= nv
        copyto!(Q[:, c], v)
    end
    return nothing
end

"""
    qr_bcgs(A; s = 64, reorth = true, rtol = 0, capacity = (2m, 2n), matmul! = mul!) -> UpdatableQR

Factor the `m` by `n` matrix `A` with `m >= n` by block classical Gram-Schmidt and return it as an
[`UpdatableQR`](@ref) holding the thin `Q` (`m` by `n`, orthonormal columns) and the upper
triangular `R`, ready for the updating verbs. Columns are orthogonalized one at a time against
the current block and the accumulated projection against earlier blocks is deferred, then
flushed every `s` columns through `matmul!(C, A, B, alpha, beta)`, which defaults to
`LinearAlgebra.mul!`. `capacity` is passed through to the returned factorization.

`reorth = true` runs the projection a second time at every flush and at every column, which
costs roughly twice as much and gives orthogonality of order `eps` provided the product of the
condition number and `eps` is well below one. `reorth = false` leaves the loss of orthogonality
growing as the square of the condition number. Either way, `R` is accumulated from the
projection coefficients rather than recomputed as `Q'*A`, so the reconstruction residual
`norm(Q*R - A)` stays at machine precision even where orthogonality has been lost; only
`norm(Q'*Q - I)` reflects `reorth`. [`qr_householder`](@ref) is more accurate than either on
orthogonality and is the recommended path; see the construction page of the documentation for
the measured ratios and residuals.

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
    mcap, ncap = Int(capacity[1]), Int(capacity[2])
    mcap >= m || throw(ArgumentError("row capacity $mcap is below the size $m"))
    ncap >= n || throw(ArgumentError("column capacity $ncap is below the size $n"))
    colnorms = Vector{T}(undef, n)
    for c in 1:n
        colnorms[c] = T(norm(view(A, :, c)))
    end
    M = Matrix{T}(undef, m, n)
    copyto!(M, A)
    # Factoring directly into the leading blocks of the capacity-sized buffers, rather than into
    # freshly sized Q and R that are copied in afterward, means every entry is written once.
    qbuf = zeros(T, mcap, ncap + 1)
    Q = view(qbuf, 1:m, 1:n)
    rbuf = zeros(T, ncap + 1, ncap + 1)
    R = view(rbuf, 1:n, 1:n)
    coef = Matrix{T}(undef, min(s, n), n)
    _qr_bcgs_kernel!(Q, R, M, colnorms, n, s, reorth, rtol, matmul!, coef)
    return _wrap_qr(qbuf, rbuf, m, n)
end

"""
    qr_bcgs!(F::UpdatableQR, A; s = 64, reorth = true, rtol = 0, matmul! = mul!) -> F

Factor the `m` by `n` matrix `A` with `m >= n` by block classical Gram-Schmidt into the existing
`F`, overwriting its own storage. `A` is destroyed: it is used directly as the algorithm's
working matrix, exactly as `LinearAlgebra.qr!` destroys its argument. Throws `ArgumentError`
naming `capacity(F)` and `size(A)` when `F`'s capacity cannot hold the result, rather than
growing it.

A caller who refactorizes repeatedly into the same `F`, at a shape no larger than `capacity(F)`
in either dimension, allocates nothing. See [`qr_bcgs`](@ref) for `s`, `reorth`, `rtol`,
`matmul!` and the `ArgumentError` thrown on a rank-deficient column.

On that `ArgumentError`, `F`'s storage has already been partially overwritten; unlike the
updating verbs, `issuccess(F)` does not reflect this, so `F` must be rebuilt rather than reused.
"""
function qr_bcgs!(
        F::UpdatableQR{T, S, <:DenseQ}, A::AbstractMatrix{T}; s::Int = 64, reorth::Bool = true,
        rtol::Real = 0, matmul!::MF = mul!
    ) where {T, S, MF}
    # `matmul!` is pinned to its own type parameter `MF`: left bare, the call into
    # `_qr_bcgs_kernel!` measurably allocates on this method's flush block (its adjoint operand
    # and multiple call sites are enough that inference no longer resolves it to a static
    # `invoke`), even though the identical call written directly (not passed through this method)
    # does not.
    Base.require_one_based_indexing(A)
    m, n = size(A)
    # lazy"..." on every message here for the same reason as in cholesky_crout!: several
    # interpolated exceptions in one method measurably allocate even on the path that never
    # throws.
    m >= n || throw(DimensionMismatch(lazy"A is $m by $n; qr_bcgs! requires m >= n"))
    s >= 1 || throw(ArgumentError(lazy"block size s must be at least 1, got $s"))
    mcap, ncap = capacity(F)
    mcap >= m || throw(ArgumentError(lazy"row capacity $mcap is below the size $m"))
    ncap >= n || throw(ArgumentError(lazy"column capacity $ncap is below the size $n"))
    colnorms = view(F.corr, 1:n)
    for c in 1:n
        colnorms[c] = T(norm(view(A, :, c)))
    end
    q = getfield(F, :qrep)
    _clear_outside!(q.buf, m, n)
    # R's within-block entries accumulate rather than overwrite (`R[j, c] += coeff`), so the
    # whole buffer, not only the region outside the new active block, must start at zero.
    fill!(F.factors, zero(T))
    Q = view(q.buf, 1:m, 1:n)
    R = view(F.factors, 1:n, 1:n)
    coef = view(F.rscratch, 1:min(s, n), 1:n)
    _qr_bcgs_kernel!(Q, R, A, colnorms, n, s, reorth, rtol, matmul!, coef)
    q.m = m
    q.n = n
    F.m = m
    F.n = n
    return F
end
