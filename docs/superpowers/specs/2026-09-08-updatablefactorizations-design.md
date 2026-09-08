# UpdatableFactorizations.jl — design

Date: 2026-09-08
Status: approved, not yet implemented

## 1. Purpose

A pure-Julia, MIT-licensed package that maintains Cholesky, LU and QR factorizations under
modification — rank-1 update and downdate, column and row insertion and deletion, and column
shifts — and that can build those factorizations itself using the blocked algorithms of
Camarero (2018).

Two things are in scope, in this order of importance:

1. **The updating layer.** The rank-1 operations (`ch1up`, `ch1dn`, `qr1up`, `lu1up`) are not
   available in any MIT pure-Julia package today. Neither is LU updating of any kind, QR row
   deletion, or the column shifts. This is the package's reason to exist.
2. **The construction layer.** Camarero's Algorithms 1–3, which produce factorizations in the
   representation the updating layer wants (explicit thin Q, directly addressable L/U/R), and
   which are blocked for element types where the standard library has no blocked path.

## 2. Prior art and the gap

| Package | License | Implementation | Covers |
| --- | --- | --- | --- |
| `qrupdate-ng` | GPL-3.0-or-later | Fortran | all 13 routines |
| `QRupdatesFast.jl` | MIT source, links GPL-3.0 `QRupdate_ng_jll` | `ccall` wrapper | all 13, via the GPL library |
| `QRupdate.jl` | MIT | pure Julia | `qraddcol`, `qrdelcol`, `qraddrow`, Q-less |
| `UpdatableQRFactorizations.jl` | MIT | pure Julia, Givens Q | column add/remove |
| `UpdatableCholeskyFactorizations.jl` | MIT | pure Julia | append column, remove column |

Coverage of `qrupdate`'s 13 routines by MIT pure-Julia code:

| routine | operation | covered |
| --- | --- | --- |
| `qr1up` | QR rank-1 update | no |
| `qrinc` | QR insert column | yes |
| `qrdec` | QR delete column | yes |
| `qrshc` | QR shift columns | no |
| `qrinr` | QR insert row | yes |
| `qrder` | QR delete row | no |
| `ch1up` | Cholesky rank-1 update | no |
| `ch1dn` | Cholesky rank-1 downdate | no |
| `chinx` | Cholesky symmetric insert | partly (append only) |
| `chdex` | Cholesky symmetric delete | yes |
| `chshx` | Cholesky symmetric shift | no |
| `lu1up` | LU rank-1 update | no |
| `lup1up` | pivoted LU rank-1 update | no |

Eight of thirteen are unserved and a ninth only partly, including every rank-1 operation. What
exists is also spread across three packages with three different Q representations and no common
interface.

`QRupdatesFast.jl` reaches all thirteen but links GPL-3.0 code, so it cannot be a dependency of
an MIT package.

## 3. Measured basis

The source paper is Camarero (2018), *Simple, Fast and Practicable Algorithms for Cholesky, LU
and QR Decomposition Using Fast Rectangular Matrix Multiplication* (arXiv:1812.02056). Its
Algorithms 1–3 are left-looking blocked factorizations whose trailing-submatrix update is
deferred and flushed every `s` columns through a fast rectangular matrix multiplication. The
paper reports 48% / 57% / 87% less time than GSL for Cholesky / LU / QR.

GSL has no blocked BLAS-3 path. Measured against one that does — LAPACK via OpenBLAS, single
threaded, on an AMD Ryzen AI 7 350 (znver5), Julia 1.12.7 — the paper's speed claim does not
hold. Ratios are relative to the blocked LAPACK routine; above 1.00 would be a win.

| algorithm | baseline | best at n=2000 | best at n=4000 |
| --- | --- | --- | --- |
| Alg 1, Cholesky | `potrf` | 0.88 | 0.93 |
| Alg 2, LU | `getrf`, pivoted | 0.95 | 0.89 |
| Alg 3, QR | `geqrf` + forming thin Q | 0.92 | 0.93 |

Three consequences shape this design:

- **Strassen is not used.** Routing Algorithm 1's deferred flush through a Strassen-Winograd
  multiply lowers it from 0.63 to 0.47 (n=4000, s=256). The flush is a rank-`s` update whose
  inner dimension is `s`, so the flop cut cannot pay for the extra buffer traffic. Strassen
  additionally requires `min(m,n,k) >= 256`, which is past the point where the `O(n^2 s)`
  per-column panel cost dominates. Speed falls monotonically in `s` at every size measured.
- **Algorithm 3 is not the default QR.** Block-CGS loses orthogonality as `kappa(A)^2`. Measured
  `norm(Q'Q - I)` is 1.2e-12 to 7.2e-12 against Householder's 6.8e-14 to 1.0e-13, and
  `norm(QR - A)/norm(A)` is 20-60x worse.
- **Algorithm 2 is worth shipping.** `lu!(A, NoPivot())` in the standard library is an unblocked
  generic fallback running at 0.05-0.06 of pivoted `getrf`. Algorithm 2 beats it by 14.7x at
  n=2000 and 17.8x at n=4000. Where the standard library has no blocked path, the paper's
  recipe is worth a great deal.

The benchmark host's CPU clock is unpinned, so these numbers are indicative rather than
gate-authoritative; the M1 benchmark gate re-runs them on a clock-locked host. The effect is
far larger than clock spread and consistent across three algorithms and two sizes.

Raw spike data and the sweep script are reproduced in `bench/` as part of M1.

## 4. Architecture

Three layers. Each is usable without the one above it.

### Layer 1 — factorization types

Mutable, subtyping `LinearAlgebra.Factorization{T}`, each carrying spare capacity so an
insertion does not reallocate:

```julia
mutable struct UpdatableCholesky{T,S<:AbstractMatrix{T}} <: Factorization{T}
    factors::S       # capacity m x m; the active factor is the leading n x n block
    n::Int
    uplo::Char
end

mutable struct UpdatableLU{T,S<:AbstractMatrix{T}} <: Factorization{T}
    L::S
    U::S
    p::Vector{Int}   # identity permutation when unpivoted
    pivoted::Bool
end

mutable struct UpdatableQR{T,S<:AbstractMatrix{T},Q<:AbstractQRep{T}} <: Factorization{T}
    Q::Q
    R::S
    m::Int
    n::Int
end
```

These implement the `Factorization` interface: `\`, `ldiv!`, `size`, `det`, `logdet`, `Matrix`,
`getproperty` for the named factors.

Constructors from standard-library factorizations — `UpdatableQR(::QRCompactWY)`,
`UpdatableCholesky(::Cholesky)`, `UpdatableLU(::LU)` — make Layer 3 usable without Layer 2.
This is deliberate: the updating layer must not depend on the paper's algorithms.

### Layer 2 — construction (Camarero Algorithms 1–3)

```julia
cholesky_crout(A; s, uplo = :L, matmul = DefaultMatMul())
lu_crout(A; s, pivot = NoPivot(), matmul = DefaultMatMul())
qr_bcgs(A; s, reorth = true, matmul = DefaultMatMul())
```

Each returns the corresponding Layer-1 type. `lu_crout` ships both the paper's unpivoted form
and a partial-pivoting variant, the latter applying row interchanges retroactively to the stored
panel. `qr_bcgs` handles `m x n` with `m >= n` (thin QR), extending the paper's square-only
statement, and `reorth = true` performs one reorthogonalization pass (BCGS2), which brings
orthogonality from `kappa^2` to `kappa` at roughly twice the cost.

A Householder QR is not written here. `qr_householder` delegates to `LinearAlgebra.qr`, or to
`PureBLAS.geqrf!` when the PureBLAS extension is loaded, and wraps the result in an
`UpdatableQR`. It is the accuracy-preferring path and the default recommendation in the docs.

The matrix multiplication used for the deferred flush sits behind a one-function seam:

```julia
abstract type AbstractMatMul end
struct DefaultMatMul <: AbstractMatMul end          # LinearAlgebra.mul!
matmul!(::AbstractMatMul, C, A, B, alpha, beta)
```

The seam exists so a generic-`T` or pure-Julia multiply can be substituted, not as a speed
claim. The PureBLAS extension supplies `PureBLASMatMul` for users who want no OpenBLAS in the
process.

### Layer 3 — updating

Eight verbs, dispatching on the factorization type, covering all thirteen `qrupdate` routines:

| verb | `UpdatableCholesky` | `UpdatableQR` | `UpdatableLU` |
| --- | --- | --- | --- |
| `update!(F, u)` / `update!(F, u, v)` | `ch1up` | `qr1up` | `lu1up`, `lup1up` |
| `downdate!(F, u)` | `ch1dn` | — | — |
| `insert_column!(F, j, x)` | `chinx` | `qrinc` | — |
| `delete_column!(F, j)` | `chdex` | `qrdec` | — |
| `insert_row!(F, i, x)` | — | `qrinr` | — |
| `delete_row!(F, i)` | — | `qrder` | — |
| `shift_columns!(F, i, j)` | `chshx` | `qrshc` | — |

For `UpdatableCholesky` the column verbs act symmetrically, inserting or deleting the matching
row as well; the factorization type makes that unambiguous and it keeps the verb set at eight.

`update!(F::UpdatableLU, u, v)` dispatches on `F.pivoted` to reach `lu1up` or `lup1up`
behavior, maintaining the permutation in the pivoted case.

### Q representations

```julia
abstract type AbstractQRep{T} end
struct DenseQ{T,S<:AbstractMatrix{T}} <: AbstractQRep{T}   # explicit thin or full Q
struct GivensQ{T} <: AbstractQRep{T}                       # sequence of Givens rotations
```

`DenseQ` is the default. `GivensQ` stores Q implicitly for large `m`, where an explicit `m x n`
Q does not fit; it trades memory for the cost of applying the rotation sequence.

## 5. Numerics and failure modes

Fail fast, always with a message naming the cause:

- `downdate!` throws when the downdated matrix would not be positive definite. This is the
  classic `ch1dn` failure and must never surface as a silent `NaN`. Hyperbolic rotations are
  used, in the mixed form of Bojanczyk et al. for stability.
- `lu_crout` with `pivot = NoPivot()` throws on a zero or numerically negligible pivot, naming
  the column. Algorithm 2 requires nonzero leading principal minors and cannot detect the
  failure any later.
- `cholesky_crout` throws on a non-positive diagonal, naming the column.
- Insertions beyond the allocated capacity grow the storage; the growth is documented, and a
  capacity argument lets a caller pre-size to avoid it.

Documented accuracy expectations: `qr_bcgs(reorth = false)` gives `kappa^2` orthogonality,
`reorth = true` gives `kappa`, and `qr_householder` gives machine-precision orthogonality. The
measured numbers from section 3 appear in the docs.

## 6. Provenance

Clean-room with respect to the GPL prior art. Algorithms are derived only from published
literature:

- Gill, Golub, Murray, Saunders, *Methods for modifying matrix factorizations*, Math. Comp. 28
  (1974) — rank-1 update and downdate.
- Daniel, Gragg, Kaufman, Stewart, *Reorthogonalization and stable algorithms for updating the
  Gram-Schmidt QR factorization*, Math. Comp. 30 (1976).
- Golub and Van Loan, *Matrix Computations*, section 6.5.
- Björck, *Numerical Methods for Least Squares Problems*, section 3.2.
- Bennett, *Triangular factors of modified matrices*, Numer. Math. 7 (1965) — LU update.
- Bojanczyk, Brent, Van Dooren, de Hoog, on stable hyperbolic-rotation downdating.
- Camarero, arXiv:1812.02056 — Algorithms 1–3.

`qrupdate-ng` source is never read. It is a benchmark target and a routine-list reference only.
`QRupdate.jl`, `UpdatableQRFactorizations.jl` and `UpdatableCholeskyFactorizations.jl` are MIT
and readable, but are used for API taste and as benchmark comparisons rather than copied, so the
package ships a single MIT notice.

This record is published, not merely kept in this spec. The documentation carries a Provenance
page stating, for every routine the package implements, which article it derives from, and
listing every non-GPL reference implementation consulted together with what it was consulted
for. The page states explicitly that `qrupdate-ng` source was not read. Each algorithm's
docstring cites its source article, so the attribution is visible from the REPL as well as from
the rendered documentation. Any reference implementation consulted after this spec is written is
added to that page in the same commit as the code that consulted it.

## 7. Verification

**TypeContracts.** `@contract` on `AbstractQRep` (`lmul!`, `rmul!`, `Matrix`, `size`) and on
`AbstractMatMul` (`matmul!`). `@invariants` on the three factorization types for the structural
properties: triangularity of the stored factors, `p` a valid permutation, `n <= capacity`.

**StrictMode.** `@assert_noalloc` and `@assert_typestable` on the rank-1 update and downdate
kernels. Those run in loops and are where a stray allocation or instability costs most.
`StrictModeTest`'s `@test_noalloc` and `@test_typestable` are the CI gate.

**Tests**, using TestItems and TestItemRunner, one `@testitem` per routine:

- Every update checked against a freshly recomputed factorization, on both
  `norm(Q*R - A)/norm(A)` and `norm(Q'Q - I)`.
- Generic indexing: entry points exercised on `OffsetArray`- and `view`-wrapped inputs, with
  output axes asserted to track input axes.
- Element types: `Float64`, `Float32`, `ComplexF64`, `BigFloat`, `ForwardDiff.Dual`.
- Failure paths: `downdate!` past positive-definiteness, zero pivot in unpivoted `lu_crout`,
  dimension mismatches — asserted on the message, not only the exception type.

## 8. Dependencies and environments

| environment | packages |
| --- | --- |
| `Project.toml` `[deps]` | `LinearAlgebra`, `StrictMode`, `TypeContracts` |
| `test/Project.toml` | `StrictModeTest`, `TestItemRunner`, `Test`, `ForwardDiff`, `OffsetArrays` |
| `bench/Project.toml` | `QRupdatesFast`, `QRupdate`, `UpdatableQRFactorizations`, `UpdatableCholeskyFactorizations`, `Chairmarks` |
| `ext/` | `PureBLAS` (weak dependency) |

`julia = "1.12"`. StrictMode 0.4.2 and TypeContracts 0.14 both require 1.12, so the usual
1.10-LTS floor is not reachable.

`QRupdatesFast` pulls in GPL-3.0 `qrupdate-ng` through its JLL. It is confined to `bench/` and
must not appear in `Project.toml` or `test/Project.toml`.

## 9. Documentation and benchmarks

DocumenterVitepress, deployed with `DocumenterVitepress.deploydocs`. Pages: home, getting
started, construction (Algorithms 1–3, carrying the section-3 table), updating and downdating,
Q representations, benchmarks, provenance, API reference. Coveralls coverage badge, MIT badge.

The provenance page carries two tables. The first maps every implemented routine to the article
it derives from, using the citations in section 6. The second lists every non-GPL reference
implementation consulted — currently `QRupdate.jl`, `UpdatableQRFactorizations.jl` and
`UpdatableCholeskyFactorizations.jl`, all MIT — with its license and what it was consulted for,
and records that `qrupdate-ng` source was not read and that `QRupdatesFast.jl` is used only as a
benchmark target because it links GPL-3.0 code.

Benchmarks are reproducible from a fresh checkout: every sample is written to
`bench/results/*.json` and plots are regenerated from the saved data rather than by re-running.
Comparisons: `QRupdatesFast` (the GPL library, via its wrapper), `QRupdate.jl`,
`UpdatableQRFactorizations.jl`, `UpdatableCholeskyFactorizations.jl`, and recomputing the
factorization from scratch — the last of these being the baseline that says whether updating
pays at all for a given size and operation.

## 10. Requirements checklist

| # | requirement | status |
| --- | --- | --- |
| R1 | MIT licensed, no GPL code in the dependency graph outside `bench/` | open |
| R2 | Camarero Algorithms 1–3 implemented | open |
| R3 | LU ships both unpivoted and partially pivoted forms | open |
| R4 | QR handles rectangular `m >= n`, with optional reorthogonalization | open |
| R5 | Householder QR path delegates to PureBLAS or LinearAlgebra, not hand-written | open |
| R6 | Pluggable matmul seam, LinearAlgebra default, PureBLAS extension | open |
| R7 | All 13 `qrupdate` routines, as 8 dispatching verbs | open |
| R8 | Implicit-Givens Q representation in addition to dense | open |
| R9 | TypeContracts contracts on the two interfaces, invariants on the types | open |
| R10 | StrictMode assertions on rank-1 kernels, StrictModeTest gating CI | open |
| R11 | TestItems / TestItemRunner, including OffsetArray and non-BLAS eltypes | open |
| R12 | DocumenterVitepress docs with the honest LAPACK comparison | open |
| R13 | Reproducible benchmarks with saved datapoints | open |
| R14 | Benchmark gate re-run on a clock-locked host | open |
| R15 | Provenance published in the docs: routine-to-article map, every non-GPL reference implementation consulted, and the statement that `qrupdate-ng` source was not read; each algorithm's docstring cites its article | open |

## 11. Non-goals

- Beating LAPACK on BLAS element types. Section 3 measures that this does not happen, and no
  documentation will claim it.
- Strassen or any subcubic multiplication in the deferred flush. Measured counterproductive.
- Sparse factorizations. `PureSparse.jl` covers sparse Cholesky and LDL', including rank-k
  update and downdate on its simplicial representation.
- Multithreading. Single-threaded throughout; the update kernels are `O(mn)` and bandwidth-bound.

## 12. Milestones

- **M0** — package skeleton, CI, Coveralls, documentation shell, contracts, Layer-1 types with
  standard-library constructors, `\` and `ldiv!`.
- **M1** — Algorithms 1–3 with the matmul seam; benchmark table reproduced on a clock-locked
  host; the unpivoted-LU result documented.
- **M2** — the thirteen routines on `DenseQ`. The headline milestone.
- **M3** — `GivensQ`.
- **M4** — benchmarks against prior art, documentation, registration in General.
