# Spike: Camarero (arXiv:1812.02056) Algorithm 1 vs blocked Cholesky

Host: AMD Ryzen AI 7 350 (znver5), Julia 1.12.7, OpenBLAS ILP64, BLAS threads = 1,
governor `powersave`, boost on. **Clock is unpinned on this host — indicative, not gate-authoritative.**

Baselines: `LinearAlgebra.cholesky!` (LAPACK potrf) and `PureBLAS.potrf!`. They tie (1.00x / 1.01x).

Deferred-flush variants: `gemm` = paper-faithful full block via `mul!`; `syrk` = lower triangle only
(half the flops, not Strassen-able); `pgemm`/`psyrk` = same through PureBLAS (Strassen above n=256).

Relative error `norm(L*L' - A)/norm(A)` is 1.3e-16..2.5e-16 for every cell; accuracy is not the issue.

## Speed relative to LAPACK potrf (higher is better; >1.00x would be a win)

    n=2000   s=64    s=256   s=512   s=1024
    gemm     0.47    0.58    0.49    0.37
    syrk     0.88    0.67    0.51    0.37
    pgemm    0.48    0.44    0.39    0.32
    psyrk    0.87    0.67    0.52    0.37

    n=4000   s=64    s=256   s=512   s=1024
    gemm     0.44    0.63    0.60    0.38
    syrk     0.93    0.78    0.65    0.39
    pgemm    0.50    0.47    0.44    0.32
    psyrk    0.93    0.79    0.65    0.39

An earlier OpenBLAS-only run gave 1.02x for (n=2000, s=64, syrk) against 0.88x here — run-to-run
spread on an unpinned clock. Every other cell agreed to within a few percent.

## Findings

1. No cell beats blocked Cholesky. Best is 0.93x (syrk, s=64, n=4000).
2. The paper's faithful full-block `gemm` update peaks at 0.63x. Halving its flops with `syrk`
   is worth 1.2-2.1x, but `syrk` cannot use a rectangular fast multiply — the two are exclusive.
3. Routing the deferred update through PureBLAS's Strassen makes it *worse*, not better
   (0.47x vs 0.63x at n=4000, s=256). The flush is a rank-s update: the inner dimension is `s`,
   so Strassen's flop cut does not pay for its buffer traffic on that shape.
4. Speed falls monotonically as `s` grows, at every n. Strassen needs min(m,n,k) >= 256, so it
   only fires at s >= 256 — exactly where the O(n^2 s) per-column panel tax has already
   overtaken whatever the multiply saves. The paper's own best setting was s=200.
5. The paper's 48%/57%/87% wins are against GSL, which has no blocked BLAS-3 path. Against a
   blocked LAPACK the structural advantage is gone: Alg 1 *is* a blocked left-looking
   factorization, which is what potrf already does.

---

# Algorithms 2 (LU) and 3 (QR), clean session

An earlier run of this sweep was contaminated: `PureBLAS.activate()` was still in effect from a
previous cell and had rerouted LAPACK. Measured on the same host, n=4000, BLAS threads = 1:

    routine   with PureBLAS.activate()   clean OpenBLAS
    getrf              2014 ms               756 ms      (2.7x slower)
    geqrf              5912 ms              1637 ms      (3.6x slower)

That run reported Alg 2 at 2.83x and Alg 3 at 4.19x. Both were artifacts of the crippled baseline
and are discarded. `potrf` was unaffected, so the Algorithm 1 table above stands. The PureBLAS
regression is a separate finding worth its own investigation.

## Alg 2 (LU) vs LAPACK getrf, clean

    n=2000   s=64    s=128   s=256   s=512      n=4000   s=64    s=128   s=256   s=512
    alg2     0.95    0.92    0.84    0.70       alg2     0.89    0.91    0.86    0.68

Relative error `norm(L*U - A)/norm(A)`: 1.6e-16..2.8e-16 (LAPACK pivoted 9.0e-17..1.0e-16).

`lu!(A, NoPivot())` runs at 0.06x / 0.05x of pivoted getrf, because stdlib's unpivoted path is an
unblocked generic fallback rather than a LAPACK call. **Alg 2 beats it by 14.7x (n=2000) and
17.8x (n=4000).** This is the one measured win in the whole spike.

## Alg 3 (QR) vs LAPACK geqrf + forming thin Q, clean

    n=2000   s=32    s=64    s=128   s=256   s=512
    alg3     0.86    0.91    0.92    0.92    0.87

    n=4000   s=32    s=64    s=128   s=256   s=512
    alg3     0.86    0.91    0.93    0.93    0.88

Forming Q is included in the baseline because Alg 3 returns it: `geqrf` alone is 196 ms / 1578 ms,
`geqrf` + `Matrix(F.Q)` is 493 ms / 3958 ms at n=2000 / 4000.

Accuracy is where Alg 3 loses badly, as block-CGS must:

    quantity                    geqrf+formQ        alg3
    norm(Q*R - A)/norm(A)       1.2e-15            2.3e-14 .. 9.7e-14   (20-60x worse)
    norm(Q'Q - I)               6.8e-14, 1.0e-13   1.2e-12 .. 7.2e-12   (~50x worse)

# Verdict

All three algorithms lose to blocked LAPACK on Float64: Cholesky 0.93-0.96x, LU 0.68-0.95x,
QR 0.86-0.93x. Strassen makes Cholesky worse, not better. QR additionally gives up ~50x in
orthogonality. The paper's premise does not survive a blocked BLAS-3 baseline.

The single real gap the spike found is unpivoted LU, where stdlib has no blocked path at all.

These figures describe the paper's algorithms as the spike implemented them, and stay true of
those. The shipped `qr_bcgs` is faster than this section reports because it builds `R` from the
projection coefficients rather than from a closing `Q'*A`, which is not part of Algorithm 3. The
measurements of what ships are in the construction section below.

# Construction layer, measured

Julia 1.12.7, OpenBLAS (`LBTConfig([ILP64] libopenblas64_.so)`), one BLAS thread, 2026-09-09.
From `bench/construct_sweep.jl`, 12 interleaved rounds per comparison, `bench/results/`.

**Speed is quoted relative to the LAPACK baseline, as everywhere above: higher is better, and
above 1.00x would be a win.** The JSON itself stores the reciprocal, candidate time over baseline
time, so a figure read straight out of `ratio_gcnet_median` means the opposite of a figure here.
Timings are recorded both as wall time and with the collection pause subtracted; they agree to
within a percent throughout this run, and the figures below are the collection-free medians.

These measure the shipped entry points, which also copy the input, allocate the factor and build
the wrapper the updating verbs need. The bare-kernel figures earlier in this file do not include
any of that, so the two are directly comparable only because the entry points no longer carry
overhead worth reporting.

## Cholesky, relative to LAPACK potrf

    n=2000   s=64    s=128   s=256
    syrk     0.92    0.84    0.72
    gemm     0.54    0.55    0.51

    n=4000   s=64    s=128   s=256
    syrk     0.92    0.85    0.81
    gemm     0.47    0.50    0.50

The bare kernel measured 0.88x and 0.93x at s=64 for the two sizes, so the shipped entry point
now performs like the standalone prototype. Both confirm the earlier finding: halving the flops
with `syrk` is worth 1.7 to 2.0 times over the paper-faithful full-block `gemm`, and speed falls
as `s` grows.

## LU, and what partial pivoting costs

    n      s     pivoted    unpivoted   unpivoted   pivoting   inter-
                 vs getrf   vs getrf    vs stdlib   costs      changes
    2000   64    0.89       0.98        14.7x       9%         2000
    2000   128   0.87       0.95        13.9x       10%        2000
    2000   256   0.80       0.87        12.7x       9%         2000
    4000   64    0.89       0.93        18.1x       4%         3999
    4000   128   0.88       0.95        17.8x       8%         3999
    4000   256   0.83       0.89        16.8x       6%         3999

Read the third column before the fourth. `lu!(A, NoPivot())` is an unblocked generic fallback
running at roughly a fifteenth of `getrf`, so beating it by 12.7 to 18.1 times says almost
nothing about this package and almost everything about that fallback. Measured against `getrf`,
which is what anyone reaching for an LU factorization actually gets, the unpivoted routine is
0.87 to 0.98 — slightly slower, like every other cell in this document.

The honest claim is therefore narrow: if you need an unpivoted factorization specifically, this
is the only blocked one available and it is about fifteen times faster than the alternative. It
is not faster than LAPACK, and quoting the 18.1x beside the Cholesky and QR figures would imply
that it is.

The pivoting cost compares pivoted against unpivoted on the same plain random matrix, which
swaps at essentially every column. The pivot search is `iamax`; what remains is the row
interchanges. The diagonally dominant fixture answers nothing here, because partial pivoting
selects the diagonal every time and performs no interchange at all.

## QR, relative to LAPACK geqrf plus forming the thin Q

    n=2000          s=32    s=64    s=128   s=256
    reorth=false    1.47    1.57    1.61    1.59
    reorth=true     0.75    0.81    0.84    0.82

    n=4000          s=32    s=64    s=128   s=256
    reorth=false    1.38    1.57    1.62    1.62
    reorth=true     0.72    0.81    0.84    0.83

`qr_bcgs` without reorthogonalization is faster than the LAPACK pair at every block size and both
sizes. It builds `R` from the projection coefficients the orthogonalization already produces,
where forming it as a closing `Q'*A` costs a further 2*m*n^2 and was 44% of the routine.
Reorthogonalization runs the projection a second time and costs about twice as much, which puts
that setting below the baseline.

Reconstruction is better than the baseline's and orthogonality is worse, which is what
Gram-Schmidt trades:

    quantity                    geqrf+formQ   bcgs reorth=false   bcgs reorth=true
    norm(Q*R - A)/norm(A)       1.2e-15       5.6e-16             6.1e-16
    norm(Q'Q - I)               1.0e-13       1.7e-12             1.6e-12

The two `reorth` settings differ little on a well-conditioned random matrix; the setting decides
orthogonality as the condition number grows, not here. Reconstruction no longer follows
orthogonality at all: `Q*R` rebuilds `A` from the coefficients that built `Q`, so the residual
holds near 1e-16 across the conditioning range while `norm(Q'Q - I)` degrades normally.

## No cell beats blocked LAPACK, except QR

`qr_bcgs` without reorthogonalization reaches 1.62x. Everything else loses: Cholesky's best is
0.92x of `potrf`, pivoted LU 0.90x of `getrf`, unpivoted LU 0.98x, and `qr_bcgs` with
reorthogonalization 0.84x. Substituting a full-block multiply for the symmetric rank-k flush
costs a further 1.7 to 1.8 times.
