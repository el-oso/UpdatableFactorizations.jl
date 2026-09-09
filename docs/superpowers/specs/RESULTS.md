# Spike: Camarero (arXiv:1812.02056) Algorithm 1 vs blocked Cholesky

Host: neuromancer (AMD Ryzen AI 7 350, znver5), Julia 1.12.7, OpenBLAS ILP64, BLAS threads = 1,
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

# Construction layer, measured

Julia 1.12.7, OpenBLAS (`LBTConfig([ILP64] libopenblas64_.so)`), one BLAS thread, 2026-09-09.
From `bench/construct_sweep.jl`, 12 interleaved rounds per comparison, `bench/results/`. Every
ratio is candidate over baseline, so below 1.00 is faster and above 1.00 is slower. Timings are
reported both as wall time and with the garbage collection pause subtracted; the two agree to
within a percent throughout this run, and the figures below are the collection-free medians.

These measure the shipped entry points, which also copy the input, allocate the factor and build
the wrapper the updating verbs need. The bare-kernel numbers earlier in this file do not include
any of that.

## LU: what partial pivoting costs

    n      s     crout rowmaximum   crout nopivot     crout nopivot     row
                 vs getrf           vs stdlib nopivot vs crout rowmax   interchanges
    2000   64    1.12               0.068             0.91              2000
    2000   128   1.15               0.072             0.91              2000
    2000   256   1.25               0.079             0.92              2000
    4000   64    1.12               0.055             0.96              3999
    4000   128   1.13               0.056             0.93              3999
    4000   256   1.20               0.059             0.94              3999

The last two columns come from the same plain random matrix, which pivots at essentially every
column. Pivoting costs 4 to 9 percent: the pivot search itself is `iamax`, and what remains is
the row interchanges. The diagonally dominant fixture answers nothing here, because partial
pivoting selects the diagonal at every column and performs no interchange at all.

Unpivoted `lu_crout` runs at 0.055 to 0.079 of `lu!(A, NoPivot())`, that is 12.7 to 18.2 times
faster. This is the one comparison the package wins, and it is a statement about the standard
library rather than about LAPACK: `lu!(A, NoPivot())` is an unblocked generic fallback with no
blocked counterpart.

## Non-BLAS element types: blocking does not help

Each blocked cell is measured against the same routine at `s >= n`, which never flushes and is
therefore the unblocked left-looking factorization.

    element type          s=16    s=64    stdlib generic
    BigFloat, n=120       2.07    1.45    1.18
    Dual{Float64,1},      1.04    1.01    0.32
    n=200

Blocking is a loss for `BigFloat` and neutral for `Dual`. The deferred flush pays only when it
can hand a block to a BLAS kernel; where `mul!` is itself a scalar loop, deferring the work adds
buffer traffic and buys nothing. For `Dual` the standard library's generic `cholesky` is about
three times faster than either, so it is the routine to use for that element type.

## No cell beats blocked LAPACK

`cholesky_crout` reaches 1.09 of `potrf` at its best block size, `lu_crout` with partial pivoting
1.12 of `getrf`, and `qr_bcgs` without reorthogonalization 1.10 of `geqrf` plus forming the thin
Q. Substituting a full-block multiply for the symmetric rank-k flush costs a further 1.8 to 2.1
times, and `reorth = true` costs about 1.5 times `reorth = false`. Nothing here is faster than
the LAPACK routine it is compared against.
