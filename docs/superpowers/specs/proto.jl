# Prototype of the M2a kernels, verified against recomputed factorizations before they go
# into the implementation plan.
using LinearAlgebra
using LinearAlgebra: givensAlgorithm

mutable struct UChol{T, S <: AbstractMatrix{T}}
    factors::S
    n::Int
    uplo::Char
    work::Vector{T}
end

function UChol(C::Cholesky{T}) where {T}
    n = size(C, 1)
    cap = 2n
    f = zeros(T, cap, cap)
    # Cholesky.factors only guarantees the stored triangle; LAPACK leaves the original matrix in
    # the other one. Copy the stored triangle alone so the unstored half is a true zero.
    if C.uplo == 'L'
        for j in 1:n, i in j:n
            f[i, j] = C.factors[i, j]
        end
    else
        for j in 1:n, i in 1:j
            f[i, j] = C.factors[i, j]
        end
    end
    return UChol{T, Matrix{T}}(f, n, C.uplo, zeros(T, cap))
end

# The active factor, always presented lower-triangular: when the storage holds U with A = U'U,
# its adjoint is the lower factor L = U' and writes through it conjugate correctly.
lowerfactor(F::UChol) =
    F.uplo == 'L' ? view(F.factors, 1:F.n, 1:F.n) : adjoint(view(F.factors, 1:F.n, 1:F.n))

Base.Matrix(F::UChol) = (L = lowerfactor(F); LowerTriangular(Matrix(L)))
reconstruct(F::UChol) = (L = Matrix(F); L * L')

# --- ch1up ---------------------------------------------------------------------------------

function lowrankupdate_chol!(F::UChol, v::AbstractVector)
    length(v) == F.n || throw(DimensionMismatch("v has length $(length(v)), factorization is $(F.n)"))
    w = view(F.work, 1:F.n)
    copyto!(w, v)
    return _ch1up!(lowerfactor(F), w)
end

function _ch1up!(L, w)
    n = size(L, 1)
    for k in 1:n
        c, s, r = givensAlgorithm(L[k, k], w[k])
        L[k, k] = r
        for i in (k + 1):n
            Lik = L[i, k]
            wi = w[i]
            L[i, k] = c * Lik + s * wi
            w[i] = -conj(s) * Lik + c * wi
        end
    end
    return L
end

# --- ch1dn: LINPACK dchdd scheme, written on the lower factor ------------------------------

function lowrankdowndate_chol!(F::UChol, v::AbstractVector)
    length(v) == F.n || throw(DimensionMismatch("v has length $(length(v)), factorization is $(F.n)"))
    w = view(F.work, 1:F.n)
    copyto!(w, v)
    return _ch1dn!(lowerfactor(F), w)
end

function _ch1dn!(L, w)
    n = size(L, 1)
    T = eltype(L)
    R = real(T)
    # p = L \ w by forward substitution
    p = w
    for i in 1:n
        acc = p[i]
        for k in 1:(i - 1)
            acc -= L[i, k] * p[k]
        end
        p[i] = acc / L[i, i]
    end
    alpha = one(R) - sum(abs2, p)
    alpha > 0 || throw(PosDefException(n))
    a = sqrt(alpha)
    cs = Vector{R}(undef, n)
    sn = Vector{T}(undef, n)
    for i in n:-1:1
        r = hypot(a, abs(p[i]))
        cs[i] = a / r
        sn[i] = p[i] / r
        a = r
    end
    for j in 1:n
        xx = zero(T)
        for i in j:-1:1
            t = cs[i] * xx + sn[i] * L[j, i]
            L[j, i] = cs[i] * L[j, i] - conj(sn[i]) * xx
            xx = t
        end
    end
    return L
end

# --- append / chinx / chdex / chshx ---------------------------------------------------------

# Append index n+1 with the new row/column `x` (length n+1, x[n+1] the new diagonal).
function append_index!(F::UChol, x::AbstractVector)
    n = F.n
    length(x) == n + 1 || throw(DimensionMismatch("x has length $(length(x)), expected $(n + 1)"))
    n + 1 <= size(F.factors, 1) || error("capacity exceeded")
    L = lowerfactor(F)
    l = Vector{eltype(F.factors)}(undef, n)
    for i in 1:n                                   # solve L l = x[1:n]
        acc = x[i]
        for k in 1:(i - 1)
            acc -= L[i, k] * l[k]
        end
        l[i] = acc / L[i, i]
    end
    d = x[n + 1] - sum(abs2, l)
    real(d) > 0 || throw(PosDefException(n + 1))
    F.n = n + 1
    Lnew = lowerfactor(F)
    for k in 1:n
        Lnew[n + 1, k] = conj(l[k])
    end
    Lnew[n + 1, n + 1] = sqrt(real(d))
    for k in 1:n
        Lnew[k, n + 1] = zero(eltype(Lnew))
    end
    return F
end

# Delete index j. Writing L in blocks about j, the leading block and the block below-left of j
# are already the factors of the reduced matrix; only the trailing block changes, and it changes
# by a rank-1 update with the deleted column's tail.
function delete_index!(F::UChol, j::Int)
    n = F.n
    1 <= j <= n || throw(BoundsError(F, j))
    L = lowerfactor(F)
    tail = [L[i, j] for i in (j + 1):n]
    for c in 1:(n - 1)                             # drop row j and column j
        for r in 1:(n - 1)
            L[r, c] = L[r + (r >= j), c + (c >= j)]
        end
    end
    F.n = n - 1
    if !isempty(tail)
        _ch1up!(view(lowerfactor(F), j:(n - 1), j:(n - 1)), tail)
    end
    return F
end

spd(n, T = Float64) = (B = randn(T, n, n); Matrix(Hermitian(B * B' + n * I)))

function check(name, got, want)
    e = norm(got - want) / norm(want)
    println(rpad(name, 34), e < 1.0e-10 ? "ok  " : "FAIL", "  relerr ", e)
    return e
end

# --- permutation of indices, via re-triangularization -------------------------------------

# Restore lower-triangular form of B by right multiplication with Givens rotations. Entries
# already zero are skipped, so a permutation that disturbs only a band costs only that band.
function _lq!(B)
    n = size(B, 1)
    for r in 1:n, c in n:-1:(r + 1)
        iszero(B[r, c]) && continue
        cc, ss, rr = givensAlgorithm(B[r, c - 1], B[r, c])
        B[r, c - 1] = rr
        B[r, c] = zero(eltype(B))
        for i in (r + 1):n
            a = B[i, c - 1]
            b = B[i, c]
            B[i, c - 1] = cc * a + ss * b
            B[i, c] = -conj(ss) * a + cc * b
        end
    end
    return B
end

function permute_indices!(F::UChol, perm::AbstractVector{Int})
    n = F.n
    L = lowerfactor(F)
    B = Matrix(L)[perm, :]
    _lq!(B)
    for c in 1:n, r in 1:n
        L[r, c] = r >= c ? B[r, c] : zero(eltype(B))
    end
    return F
end

cyclicperm(n, i, j) = i <= j ? [1:(i - 1); (i + 1):j; i; (j + 1):n] : [1:(j - 1); i; j:(i - 1); (i + 1):n]

shift_indices!(F::UChol, i::Int, j::Int) = permute_indices!(F, cyclicperm(F.n, i, j))

# x is the new row/column in the resulting (n+1)-indexing, with x[j] the new diagonal. Append it
# at the end and then move index n+1 into position j.
function insert_index!(F::UChol, j::Int, x::AbstractVector)
    n = F.n
    length(x) == n + 1 || throw(DimensionMismatch("x has length $(length(x)), expected $(n + 1)"))
    1 <= j <= n + 1 || throw(BoundsError(F, j))
    append_index!(F, vcat(x[1:(j - 1)], x[(j + 1):(n + 1)], x[j]))
    return j == n + 1 ? F : shift_indices!(F, n + 1, j)
end

# --- Bennett's LU rank-1 update, on the LDU form -------------------------------------------

function bennett!(L, d, U, w, z, sigma)
    n = length(d)
    s = sigma
    for k in 1:n
        wk = w[k]
        zk = z[k]
        dnew = d[k] + s * wk * zk
        iszero(dnew) && error("breakdown at $k")
        alpha = s * zk / dnew
        beta = s * wk / dnew
        for i in (k + 1):n
            w[i] -= wk * L[i, k]
            L[i, k] += alpha * w[i]
        end
        for j in (k + 1):n
            z[j] -= zk * U[k, j]
            U[k, j] += beta * z[j]
        end
        s = s * d[k] / dnew
        d[k] = dnew
    end
    return L, d, U
end
