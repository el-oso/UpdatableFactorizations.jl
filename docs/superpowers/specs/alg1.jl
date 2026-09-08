# throwaway spike: Camarero 2018 (arXiv:1812.02056) Algorithm 1 vs LAPACK/PureBLAS Cholesky.
using LinearAlgebra, BenchmarkTools, Printf, PureBLAS

# Algorithm 1: column-at-a-time Crout, trailing update A -= R*R' deferred and flushed every `s`
# columns. `update` selects how the deferred flush is applied:
#   :gemm  - mul!(trailing, R, R', -1, 1)   (the paper's "fast rectangular matmul"; Strassen-able)
#   :syrk  - BLAS.syrk! on the lower triangle only (half the flops, no Strassen)
# The per-column panel is expressed as a gemv rather than the paper's scalar double loop; same
# math, different summation order, and it is what any real implementation would do.
function chol_cam!(A::AbstractMatrix{T}, s::Int; update::Symbol = :gemm) where {T}
    Base.require_one_based_indexing(A)
    n = size(A, 1)
    L = zeros(T, n, n)
    z = 1
    @views for c in 1:n
        if c == z + s
            R = L[c:n, z:(c - 1)]
            trail = A[c:n, c:n]
            if update === :gemm
                mul!(trail, R, R', -one(T), one(T))
            elseif update === :syrk
                BLAS.syrk!('L', 'N', -one(T), R, one(T), trail)
            elseif update === :pgemm
                PureBLAS.gemm!(trail, R, R'; alpha = -one(T), beta = one(T))
            else
                PureBLAS.syrk!(trail, R; uplo = 'L', alpha = -one(T), beta = one(T))
            end
            z = c
        end
        P = L[c, z:(c - 1)]                      # already-computed panel entries of row c
        acc = A[c, c] - dot(P, P)
        acc > zero(T) || error("not positive definite at column $c (pivot $acc)")
        Lcc = sqrt(acc)
        L[c, c] = Lcc
        if c < n
            col = L[(c + 1):n, c]
            copyto!(col, A[(c + 1):n, c])
            c > z && mul!(col, L[(c + 1):n, z:(c - 1)], P, -one(T), one(T))
            col ./= Lcc
        end
    end
    return L
end

spd(n) = (B = randn(n, n); Matrix(Symmetric(B * B' + n * I)))

relerr(L, A0) = norm(L * L' - A0) / norm(A0)

function sweep(ns, ss; label)
    println("\n=== $label ===")
    @printf("%6s %6s %-8s %10s %10s %8s\n", "n", "s", "variant", "ms", "vs lapack", "relerr")
    for n in ns
        A0 = spd(n)
        tl = @belapsed cholesky!(Symmetric(B, :L)) setup = (B = copy($A0)) evals = 1
        @printf(
            "%6d %6s %-8s %10.2f %10s %8.1e\n", n, "-", "lapack", 1.0e3tl, "1.00x",
            relerr(cholesky(Symmetric(A0, :L)).L, A0)
        )
        for s in ss
            s < n || continue
            for up in (:gemm, :syrk)
                L = chol_cam!(copy(A0), s; update = up)
                e = relerr(L, A0)
                t = @belapsed chol_cam!(B, $s; update = $up) setup = (B = copy($A0)) evals = 1
                @printf("%6d %6d %-8s %10.2f %9.2fx %8.1e\n", n, s, up, 1.0e3t, tl / t, e)
            end
        end
    end
    return
end
