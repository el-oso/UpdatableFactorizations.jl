# UpdatableFactorizations.jl — design

Date: 2026-09-08
Status: approved, with two open decisions marked in section 10

## 1. Purpose

A pure-Julia, MIT-licensed package that maintains Cholesky, LU and QR factorizations under
modification — rank-1 update and downdate, column and row insertion and deletion, and column
shifts — and that can build those factorizations itself using the blocked algorithms of
Camarero (2018).

Two things are in scope, in this order of importance:

1. **The updating layer.** Six of `qrupdate`'s thirteen operations have no pure-Julia
   implementation at all: QR rank-1 update, QR row deletion, both column shifts, and all LU
   updating. A seventh, symmetric Cholesky insertion, exists only for appending at the end.
   Nothing provides one interface across all three factorizations, and nothing manages capacity
   so that a sequence of insertions does not reallocate.
2. **The construction layer.** Camarero's Algorithms 1–3, which produce factorizations in the
   representation the updating layer wants — explicit thin Q, directly addressable L, U and R —
   and which give a blocked unpivoted LU where the standard library has only an unblocked one.

Cholesky rank-1 update and downdate are explicitly **not** a reason for this package to exist:
`LinearAlgebra.lowrankupdate!` and `lowrankdowndate!` already ship in the standard library.
The package extends those two functions rather than competing with them.

## 2. Prior art and the gap

| Source | License | Implementation | Covers |
| --- | --- | --- | --- |
| `LinearAlgebra` stdlib | MIT | pure Julia | `lowrankupdate!`, `lowrankdowndate!` on `Cholesky` |
| `qrupdate-ng` | GPL-3.0-or-later | Fortran | all 13 routines |
| `QRupdatesFast.jl` | MIT source, links GPL-3.0 `QRupdate_ng_jll` | `ccall` wrapper | all 13, via the GPL library |
| `QRupdate.jl` | MIT | pure Julia | `qraddcol`, `qrdelcol`, `qraddrow`, Q-less |
| `UpdatableQRFactorizations.jl` | MIT | pure Julia, Givens Q | column add/remove |
| `UpdatableCholeskyFactorizations.jl` | MIT | pure Julia | append column, remove column |

Coverage of `qrupdate`'s 13 routines by non-GPL pure-Julia code:

| routine | operation | covered |
| --- | --- | --- |
| `qr1up` | QR rank-1 update | no |
| `qrinc` | QR insert column | yes |
| `qrdec` | QR delete column | yes |
| `qrshc` | QR shift columns | no |
| `qrinr` | QR insert row | yes, Q-less only |
| `qrder` | QR delete row | no |
| `ch1up` | Cholesky rank-1 update | yes, stdlib |
| `ch1dn` | Cholesky rank-1 downdate | yes, stdlib |
| `chinx` | Cholesky symmetric insert | partly, append only |
| `chdex` | Cholesky symmetric delete | yes |
| `chshx` | Cholesky symmetric shift | no |
| `lu1up` | LU rank-1 update | no |
| `lup1up` | pivoted LU rank-1 update | no |

Six are unserved and a seventh only partly. What exists is spread across the standard library
and three packages, with three different Q representations and no common interface.

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
hold. Ratios are relative to the blocked LAPACK routine; above 1.00 would be a win. Best cell
over the swept `s` at each size.

| algorithm | flush | baseline | n=2000 | n=4000 |
| --- | --- | --- | --- | --- |
| Alg 1, Cholesky | `gemm`, as written in the paper | `potrf` | 0.58 | 0.63 |
| Alg 1, Cholesky | `syrk`, lower triangle only | `potrf` | 0.88 | 0.93 |
| Alg 2, LU, unpivoted | `gemm` | `getrf`, pivoted | 0.95 | 0.91 |
| Alg 3, QR | `gemm` | `geqrf` + forming thin Q | 0.92 | 0.93 |

The two Algorithm 1 rows differ because only the lower triangle of the trailing block is ever
read. Halving the flops with a symmetric rank-k update is worth 1.2–2.1x, but a rank-k update
cannot be expressed as a rectangular matrix multiplication, so the two are mutually exclusive.
Section 4 resolves this in favor of the rank-k form.

Three further consequences shape this design:

- **Strassen is not used.** Routing Algorithm 1's deferred flush through a Strassen-Winograd
  multiply lowers it from 0.63 to 0.47 at n=4000, s=256. The flush is a rank-`s` update whose
  inner dimension is `s`, so the flop cut cannot pay for the extra buffer traffic. Strassen
  additionally requires `min(m,n,k) >= 256`, which is past the point where the `O(n^2 s)`
  per-column panel cost dominates. Speed falls monotonically in `s` at every size measured.
- **Algorithm 3 is not the recommended QR.** Block-CGS loses orthogonality as `kappa(A)^2`.
  Measured `norm(Q'Q - I)` is 1.2e-12 to 7.2e-12 against Householder's 6.8e-14 to 1.0e-13, and
  `norm(QR - A)/norm(A)` is 20–60x worse.
- **Algorithm 2 is worth shipping.** `lu!(A, NoPivot())` in the standard library is an unblocked
  generic fallback running at 0.05–0.06 of pivoted `getrf`. Algorithm 2 beats it by 14.7x at
  n=2000 and 17.8x at n=4000. Where the standard library has no blocked path, the paper's
  recipe is worth a great deal.

Two claims are **not** measured and must not be stated as fact until they are:

- That blocking helps for non-BLAS element types. Generic `mul!` for `BigFloat` and
  `ForwardDiff.Dual` is itself a scalar triple loop with modest tiling, so the deferred flush
  may buy nothing. The M1 gate adds one `BigFloat` and one `Dual` cell to settle it.
- That pivoted `lu_crout` retains unpivoted `lu_crout`'s ratio. The sweep covers the unpivoted
  form only. Row interchanges keep the flush a rank-`s` GEMM, so it composes, but the cost was
  never timed. The M1 gate adds it.

The benchmark host's CPU clock is unpinned, so these numbers are indicative rather than
gate-authoritative; the M1 gate re-runs them on a clock-locked host. The effect is far larger
than clock spread and consistent across three algorithms and two sizes.

The spike scripts and raw sweep output are `alg1.jl`, `alg23.jl` and `RESULTS.md` alongside this
document, and are reproduced under `bench/` as part of M1.

## 4. Architecture

Three layers. Layer 3 does not depend on Layer 2.

### Layer 1 — factorization types

Mutable, subtyping `LinearAlgebra.Factorization{T}`. The types whose verbs resize carry spare
capacity so an insertion does not reallocate; `UpdatableLU` has no resizing verb and so has
none.

```julia
mutable struct UpdatableCholesky{T,S<:AbstractMatrix{T}} <: Factorization{T}
    factors::S       # cap x cap; the active factor is the leading n x n block
    n::Int
    uplo::Char
    work::Vector{T}  # rotation and projection scratch, length >= cap
end

mutable struct UpdatableLU{T,S<:AbstractMatrix{T}} <: Factorization{T}
    L::S
    U::S
    p::Vector{Int}   # identity permutation when unpivoted
    pivoted::Bool
    work::Vector{T}
end

mutable struct UpdatableQR{T,S<:AbstractMatrix{T},Q<:AbstractQRep{T}} <: Factorization{T}
    Q::Q             # active region is the leading m x n block of an mcap x ncap buffer
    R::S             # active region is the leading n x n block of an ncap x ncap buffer
    m::Int
    n::Int
    work::Matrix{T}  # scratch for Q'u, residuals, and the augmented column of section 4.4
end
```

Capacity is explicit. `UpdatableQR` is built with `capacity = (mcap, ncap)`, defaulting to
`(2m, 2n)`; both dimensions grow by doubling when an insertion exceeds them. `UpdatableCholesky`
takes `capacity = cap`, defaulting to `2n`. The active region is always the leading block, and
every kernel operates on `view`s of it.

The `work` fields exist so the rank-1 kernels can be allocation-free; see section 7.

These implement the `Factorization` interface: `\`, `ldiv!`, `size`, `det`, `logdet`, `Matrix`,
and `getproperty` for the named factors.

Constructors from standard-library factorizations — `UpdatableQR(::QRCompactWY)`,
`UpdatableCholesky(::Cholesky)`, `UpdatableLU(::LU)` — make Layer 3 usable without Layer 2.

### Layer 2 — construction (Camarero Algorithms 1–3)

```julia
cholesky_crout(A; s, uplo = :L, rankk! = default_rankk!)
lu_crout(A; s, pivot = NoPivot(), matmul! = mul!)
qr_bcgs(A; s, reorth = true, matmul! = mul!)
```

Each returns the corresponding Layer-1 type.

The deferred flush is substitutable, but it is **two** operations, not one, because Algorithm 1's
flush is symmetric and Algorithms 2 and 3's are not:

- `matmul!(C, A, B, alpha, beta)`, defaulting to `LinearAlgebra.mul!`, for Algorithms 2 and 3.
- `rankk!(C, A, alpha, beta; uplo)`, defaulting to `BLAS.syrk!` on BLAS element types and a
  lower-triangle `mul!` otherwise, for Algorithm 1.

Collapsing these into one rectangular-multiply seam would force Algorithm 1 onto the `gemm` row
of section 3 and cost it 1.2–2.1x. Both default to plain stdlib functions passed as keyword
arguments; there is no abstract type and no contract over them.

`lu_crout` ships both the paper's unpivoted form and a partial-pivoting variant. The pivoting
protocol is: after column `c` of `L` is fully formed, select the pivot row `p` by maximum
magnitude over `L[c:n, c]`, then interchange rows `p` and `c` in all of `L[:, 1:c]`, in the
not-yet-flushed columns `A[:, c+1:n]`, and in the permutation vector. Rows of `U` are untouched.
Interchanges leave the deferred flush a rank-`s` GEMM, so the block structure is preserved.

`qr_bcgs` handles `m x n` with `m >= n` (thin QR), extending the paper's square-only statement.
`reorth = true` performs one reorthogonalization pass, which delivers `O(eps)` orthogonality
provided `kappa(A) * eps` is well below one, at roughly twice the cost; `reorth = false` leaves
the `kappa^2` behavior measured in section 3.

A Householder QR is not written here. `qr_householder` delegates to `LinearAlgebra.qr` and wraps
the result in an `UpdatableQR`. It is the accuracy-preferring path and the default
recommendation in the documentation.

### Layer 3 — updating

Seven verbs covering all thirteen `qrupdate` routines. The Cholesky rank-1 pair extends the
existing standard-library functions rather than introducing new names; `update!` in particular
is avoided because it collides with common exports from `Optimisers.jl` and `Flux.jl`.

| verb | `UpdatableCholesky` | `UpdatableQR` | `UpdatableLU` |
| --- | --- | --- | --- |
| `lowrankupdate!(F, v)` | `ch1up` (extends stdlib) | — | — |
| `lowrankdowndate!(F, v)` | `ch1dn` (extends stdlib) | — | — |
| `lowrankupdate!(F, u, v)` | — | `qr1up` | `lu1up`, `lup1up` |
| `insert_column!(F, j, x)` | `chinx` | `qrinc` | — |
| `delete_column!(F, j)` | `chdex` | `qrdec` | — |
| `insert_row!(F, i, x)` | — | `qrinr` | — |
| `delete_row!(F, i)` | — | `qrder` | — |
| `shift_columns!(F, i, j)` | `chshx` | `qrshc` | — |

The two-argument `lowrankupdate!(F, v)` means `A + v*v'` and matches the stdlib signature; the
three-argument `lowrankupdate!(F, u, v)` means `A + u*v'`. For `UpdatableCholesky` the column
verbs act symmetrically, inserting or deleting the matching row as well; the factorization type
makes that unambiguous.

`lowrankupdate!(F::UpdatableLU, u, v)` dispatches on `F.pivoted` to reach `lu1up` or `lup1up`
behavior, maintaining the permutation in the pivoted case.

### Q representations

```julia
abstract type AbstractQRep{T} end
struct DenseQ{T,S<:AbstractMatrix{T}} <: AbstractQRep{T}   # explicit thin Q, m x n
struct GivensQ{T} <: AbstractQRep{T}                       # sequence of Givens rotations
```

`DenseQ` is **thin only**: `m x n` with `m >= n`. A full `m x m` Q is not offered; nothing in
Layer 3 needs it, and it is the representation whose memory cost motivates the alternative.

Thin Q constrains two verbs, and the spec is explicit about both:

- `delete_row!(F, i)` on a thin Q requires the augmented-column construction of Daniel, Gragg,
  Kaufman and Stewart: the deleted row's contribution is removed using
  `gamma = sqrt(1 - norm(Q'[:, i])^2)`. When `gamma` is at or near zero the deleted row has
  leverage one and the numerical rank drops, so `delete_row!` throws with a message naming the
  row rather than returning a factorization of something else.
- Any verb that would leave `m < n` — `delete_row!` or `insert_column!` when `m == n` — throws,
  because the type admits only `m >= n`.

`GivensQ` does **not** save memory over thin `DenseQ`: both are `O(mn)`. Its purpose is to avoid
paying to form Q at all, which section 3 measures at roughly 2.5x the cost of `geqrf` alone. It
stores the rotation sequence and applies it on demand. The cost that grows is application: every
verb appends `O(n)` to `O(m)` rotations, so apply time rises with the number of updates. The
representation therefore carries a compaction policy — materialize to a `DenseQ` and restart the
log once the log length exceeds a documented multiple of `n` — and `materialize(::GivensQ)` is
public so a caller can force it. M3 specifies which verbs `GivensQ` supports; the presumption is
all seven, and any it cannot support is documented rather than silently slow.

## 5. Numerics and failure modes

Fail fast, always with a message naming the cause:

- `lowrankdowndate!` throws `PosDefException` when the downdated matrix would not be positive
  definite, matching the standard library's behavior for the same operation. This must never
  surface as a silent `NaN`. Hyperbolic rotations are used, in the mixed form of Bojanczyk,
  Brent, Van Dooren and de Hoog for stability.
- `lu_crout` with `pivot = NoPivot()` throws `ZeroPivotException(c)` naming the column.
  "Negligible" means exactly zero by default; an `rtol` keyword widens it to a relative
  threshold against the column norm. Algorithm 2 requires nonzero leading principal minors and
  cannot detect the failure any later.
- `cholesky_crout` throws `PosDefException(c)` on a non-positive diagonal.
- `delete_row!` throws when the row has leverage one, per section 4.
- Verbs that would violate `m >= n` throw.
- Insertions beyond the allocated capacity grow the storage by doubling; the growth is
  documented, and the `capacity` argument lets a caller pre-size to avoid it.

Documented accuracy expectations: `qr_bcgs(reorth = false)` gives `kappa^2` orthogonality,
`reorth = true` gives `O(eps)` under `kappa * eps << 1`, and `qr_householder` gives
machine-precision orthogonality unconditionally. The measured numbers from section 3 appear in
the documentation.

## 6. Provenance

Clean-room with respect to the GPL prior art. Algorithms are derived only from published
literature:

- Gill, Golub, Murray and Saunders, *Methods for modifying matrix factorizations*,
  Mathematics of Computation 28 (1974), 505–535 — rank-1 update and downdate.
- Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating
  the Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772–795 — column
  and row insertion and deletion, and the augmented-column construction of section 4.
- Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5. (In the 3rd edition the
  corresponding material is section 12.5.)
- Björck, *Numerical Methods for Least Squares Problems*, section 3.2.
- Bennett, *Triangular factors of modified matrices*, Numerische Mathematik 7 (1965), 217–221 —
  unpivoted LU rank-1 update.
- Stange, Griewank and Bollhöfer, *On the efficient update of rectangular LU-factorizations
  subject to low rank modifications*, ETNA 26 (2007), 161–177 — pivoted LU rank-1 update.
- Bojanczyk, Brent, Van Dooren and de Hoog, *A note on downdating the Cholesky factorization*,
  SIAM Journal on Scientific and Statistical Computing 8 (1987), 210–221.
- Giraud, Langou and Rozložník, *The loss of orthogonality in the Gram-Schmidt orthogonalization
  process*, Computers and Mathematics with Applications 50 (2005), 1069–1075 — the `O(eps)`
  bound for one reorthogonalization pass.
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

**TypeContracts.** `@contract` on `AbstractQRep`, covering `lmul!`, `rmul!`, `Matrix`, `size`
and `materialize`. `@invariants` on the three factorization types for the structural properties:
triangularity of the stored factors, `p` a valid permutation, `n <= cap`, `m >= n`. There is no
contract over the flush functions — they are plain keyword arguments defaulting to stdlib.

**StrictMode.** `@assert_noalloc` and `@assert_typestable` on the rank-1 update and downdate
kernels **for `DenseQ` only**. Those kernels run in loops and are where a stray allocation or
instability costs most; the `work` fields of section 4 are what make the no-allocation
assertion satisfiable. `GivensQ` appends to a growing rotation log and therefore carries
`@assert_typestable` alone. `StrictModeTest`'s `@test_noalloc` and `@test_typestable` are the
CI gate.

**Tests**, using TestItems and TestItemRunner, one `@testitem` per routine:

- Every update checked against a freshly recomputed factorization, on both
  `norm(Q*R - A)/norm(A)` and `norm(Q'Q - I)`.
- Cholesky rank-1 update and downdate checked against `LinearAlgebra.lowrankupdate!` and
  `lowrankdowndate!` directly, since the package extends those functions.
- Generic indexing: entry points exercised on `OffsetArray`- and `view`-wrapped inputs, with
  output axes asserted to track input axes.
- Element types: `Float64`, `Float32`, `ComplexF64`, `BigFloat`, `ForwardDiff.Dual`.
- Failure paths: downdate past positive-definiteness, zero pivot in unpivoted `lu_crout`,
  `delete_row!` on a leverage-one row, verbs that would make `m < n`, dimension mismatches —
  asserted on the message, not only the exception type.

## 8. Dependencies and environments

| environment | packages |
| --- | --- |
| `Project.toml` `[deps]` | `LinearAlgebra`, `StrictMode`, `TypeContracts` |
| `test/Project.toml` | `StrictModeTest`, `TestItemRunner`, `Test`, `ForwardDiff`, `OffsetArrays` |
| `bench/Project.toml` | `QRupdatesFast`, `QRupdate`, `UpdatableQRFactorizations`, `UpdatableCholeskyFactorizations`, `Chairmarks` |

Compat: `julia = "1.12"`, `StrictMode = "0.4.1"`, `TypeContracts = "0.14"`. StrictMode and
TypeContracts both require 1.12, so the usual 1.10-LTS floor is not reachable. StrictModeTest
0.4.1 in the test environment.

`QRupdatesFast` pulls in GPL-3.0 `qrupdate-ng` through its JLL. It is confined to `bench/` and
must not appear in `Project.toml` or `test/Project.toml`.

Registration in General (M4) is gated on `StrictMode`, `StrictModeTest` and `TypeContracts`
being registered — all three are — and on any weak dependency being registered too. `PureBLAS`
is **not** in General, which is one of the two reasons section 10 proposes dropping the PureBLAS
extension.

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
Comparisons: `LinearAlgebra.lowrankupdate!`/`lowrankdowndate!` for the Cholesky rank-1 pair,
`QRupdatesFast` (the GPL library, via its wrapper) for everything it covers, `QRupdate.jl`,
`UpdatableQRFactorizations.jl`, `UpdatableCholeskyFactorizations.jl`, and recomputing the
factorization from scratch — the last being the baseline that says whether updating pays at all
for a given size and operation.

## 10. Requirements checklist

| # | requirement | status |
| --- | --- | --- |
| R1 | MIT licensed, no GPL code in the dependency graph outside `bench/` | open |
| R2 | Camarero Algorithms 1–3 implemented | open |
| R3 | LU ships both unpivoted and partially pivoted forms, pivoting per section 4 | open |
| R4 | QR handles rectangular `m >= n`, with optional reorthogonalization | open |
| R5 | Householder QR path delegates to `LinearAlgebra.qr`, not hand-written | open |
| R6 | Substitutable flush: `matmul!` and `rankk!` keyword arguments | **decision D1** |
| R7 | All 13 `qrupdate` routines, as 7 dispatching verbs, extending stdlib where it exists | open |
| R8 | Implicit-Givens Q representation with a documented compaction policy | open |
| R9 | TypeContracts contract on `AbstractQRep`, invariants on the types | open |
| R10 | StrictMode assertions on `DenseQ` rank-1 kernels, StrictModeTest gating CI | open |
| R11 | TestItems / TestItemRunner, including OffsetArray and non-BLAS eltypes | open |
| R12 | DocumenterVitepress docs with the honest LAPACK comparison | open |
| R13 | Reproducible benchmarks with saved datapoints | open |
| R14 | M1 gate on a clock-locked host, including pivoted LU and non-BLAS eltype cells | open |
| R15 | Provenance published in the docs, and cited in every algorithm's docstring | open |
| R16 | Milestone order | **decision D2** |

**D1 — the PureBLAS extension.** The approved design had a pluggable matmul behind an
`AbstractMatMul` type with a contract, plus a PureBLAS extension. Two measurements since argue
against it: `PureBLAS.activate()` makes `getrf` 2.7x and `geqrf` 3.6x slower than OpenBLAS, and
PureBLAS is not registered in General, so it cannot be a registered weak dependency. This spec
therefore proposes plain `matmul!` and `rankk!` keyword arguments defaulting to stdlib, with no
abstract type, no contract and no extension. The substitutability the approved design asked for
is preserved; only the machinery is dropped. **Requires confirmation.**

**D2 — milestone order.** The approved order was M0, M1, M2, M3, M4. Section 4 states Layer 3
does not depend on Layer 2, and M0 already provides Layer-1 types with standard-library
constructors, so M1 and M2 are independent. Running M2 before M1 lets the updating layer
discover what it needs from the representation — workspace fields, capacity growth, thin-Q row
deletion — before Layer 2 is written to produce it. This spec therefore proposes M0, M2, M1,
M3, M4, with M2 split into a Cholesky-and-LU part (no resizing) and a QR part.
**Requires confirmation.**

## 11. Non-goals

- Beating LAPACK on BLAS element types. Section 3 measures that this does not happen, and no
  documentation will claim it.
- Strassen or any subcubic multiplication in the deferred flush. Measured counterproductive.
- Competing with `LinearAlgebra.lowrankupdate!` and `lowrankdowndate!`. The package extends
  them.
- A full `m x m` dense Q. Thin only.
- Sparse factorizations. `PureSparse.jl` covers sparse Cholesky and LDL', including rank-k
  update and downdate on its simplicial representation.
- Multithreading. Single-threaded throughout.

## 12. Milestones

Order subject to decision D2. Under the proposed order:

- **M0** — package skeleton, CI, Coveralls, documentation shell, contracts, Layer-1 types with
  capacity and workspace, standard-library constructors, `\` and `ldiv!`.
- **M2a** — Cholesky and LU updating: the stdlib-extending rank-1 pair, symmetric insert,
  delete and shift, and both LU rank-1 forms. No resizing of Q.
- **M2b** — QR updating on `DenseQ`: rank-1 update, column insert, delete and shift, row insert
  and delete including the leverage-one failure path.
- **M1** — Algorithms 1–3 with the `matmul!` and `rankk!` seam; the gate of R14 on a
  clock-locked host; the unpivoted-LU result documented.
- **M3** — `GivensQ`, with its compaction policy and its supported-verb table.
- **M4** — benchmarks against prior art, documentation including the provenance page,
  registration in General.
