# throwaway spike: Camarero 2018 Algorithms 2 (LU) and 3 (QR) vs LAPACK.
using LinearAlgebra, BenchmarkTools, Printf

# Algorithm 2, unpivoted Crout LU with the trailing update deferred and flushed every `s` columns.
# A is overwritten. Returns (L, U) with U unit upper triangular, as the paper defines them.
function lu_cam!(A::AbstractMatrix{T}, s::Int) where {T}
    Base.require_one_based_indexing(A)
    n = size(A, 1)
    L = zeros(T, n, n)
    U = Matrix{T}(I, n, n)
    z = 1
    @views for c in 1:n
        if c == z + s
            mul!(A[c:n, c:n], L[c:n, z:(c - 1)], U[z:(c - 1), c:n], -one(T), one(T))
            z = c
        end
        col = L[c:n, c]
        copyto!(col, A[c:n, c])
        c > z && mul!(col, L[c:n, z:(c - 1)], U[z:(c - 1), c], -one(T), one(T))
        Lcc = L[c, c]
        iszero(Lcc) && error("zero pivot at column $c; Algorithm 2 has no pivoting")
        if c < n
            row = U[c, (c + 1):n]
            copyto!(row, A[c, (c + 1):n])
            c > z && mul!(row, transpose(U[z:(c - 1), (c + 1):n]), L[c, z:(c - 1)], -one(T), one(T))
            row ./= Lcc
        end
    end
    return L, U
end

# Algorithm 3, block-CGS QR for m x n with m >= n. Returns (Q, R), Q explicit and thin (m x n).
function qr_cam(A::AbstractMatrix{T}, s::Int) where {T}
    Base.require_one_based_indexing(A)
    m, n = size(A)
    M = copy(A)
    Q = zeros(T, m, n)
    z = 1
    @views for c in 1:n
        if c == z + s
            BQ = Q[:, z:(c - 1)]
            BM = M[:, c:n]
            C = BQ' * BM
            mul!(BM, BQ, C, -one(T), one(T))
            z = c
        end
        v = M[:, c]
        for j in z:(c - 1)
            u = Q[:, j]
            axpy!(-dot(u, v), u, v)
        end
        v ./= norm(v)
        copyto!(Q[:, c], v)
    end
    return Q, Q' * A
end

function lu_sweep(ns, ss)
    println("\n=== LU: Alg 2 vs LAPACK getrf ===")
    @printf("%6s %6s %-12s %10s %10s %9s\n", "n", "s", "variant", "ms", "vs pivoted", "relerr")
    for n in ns
        A0 = randn(n, n) + n * I |> Matrix          # diagonally dominant: unpivoted LU is safe
        tp = @belapsed lu!(B) setup = (B = copy($A0)) evals = 1
        tn = @belapsed lu!(B, NoPivot()) setup = (B = copy($A0)) evals = 1
        @printf(
            "%6d %6s %-12s %10.2f %9.2fx %9.1e\n", n, "-", "lapack piv", 1.0e3tp, 1.0,
            norm(Matrix(lu(A0)) - A0) / norm(A0)
        )
        @printf(
            "%6d %6s %-12s %10.2f %9.2fx %9.1e\n", n, "-", "lapack nopiv", 1.0e3tn, tp / tn,
            norm(Matrix(lu(A0, NoPivot())) - A0) / norm(A0)
        )
        for s in ss
            s < n || continue
            L, U = lu_cam!(copy(A0), s)
            e = norm(L * U - A0) / norm(A0)
            t = @belapsed lu_cam!(B, $s) setup = (B = copy($A0)) evals = 1
            @printf("%6d %6d %-12s %10.2f %9.2fx %9.1e\n", n, s, "alg2", 1.0e3t, tp / t, e)
        end
    end
    return
end

function qr_sweep(ns, ss)
    println("\n=== QR: Alg 3 vs LAPACK geqrf (+ forming thin Q, which Alg 3 also returns) ===")
    @printf(
        "%6s %6s %-12s %10s %10s %9s %9s\n", "n", "s", "variant", "ms", "vs geqrf+Q",
        "relerr", "ortherr"
    )
    for n in ns
        A0 = randn(n, n)
        tf = @belapsed qr!(B) setup = (B = copy($A0)) evals = 1
        tq = @belapsed (F = qr!(B); Matrix(F.Q)) setup = (B = copy($A0)) evals = 1
        F = qr(A0); Qh = Matrix(F.Q)
        @printf(
            "%6d %6s %-12s %10.2f %9s %9.1e %9.1e\n", n, "-", "geqrf only", 1.0e3tf, "-",
            norm(Qh * F.R - A0) / norm(A0), norm(Qh' * Qh - I)
        )
        @printf(
            "%6d %6s %-12s %10.2f %9.2fx %9.1e %9.1e\n", n, "-", "geqrf+formQ", 1.0e3tq, 1.0,
            norm(Qh * F.R - A0) / norm(A0), norm(Qh' * Qh - I)
        )
        for s in ss
            s < n || continue
            Q, R = qr_cam(A0, s)
            t = @belapsed qr_cam($A0, $s) evals = 1
            @printf(
                "%6d %6d %-12s %10.2f %9.2fx %9.1e %9.1e\n", n, s, "alg3", 1.0e3t, tq / t,
                norm(Q * R - A0) / norm(A0), norm(Q' * Q - I)
            )
        end
    end
    return
end
