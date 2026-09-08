# M2b: QR updating on a dense Q — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `UpdatableQR` with an explicit thin orthonormal factor, supporting all six QR modifications of `qrupdate`'s catalogue — rank-1 update, column insert, delete and shift, row insert and delete — with the leverage-one deletion failure path that no pure-Julia package currently gets right, and `qr_householder`, the accuracy-preferring construction path that delegates to `LinearAlgebra.qr`.

**Architecture:** One mutable `LinearAlgebra.Factorization` subtype holds `R` in an over-allocated square buffer and delegates the orthonormal factor to an `AbstractQRep` behind a five-operation contract; `DenseQ` is the one implementation, storing the thin factor explicitly in an over-allocated buffer. Both buffers carry one spare row and column beyond the user-visible capacity, because every verb works in a basis one column wider than the factorization: the spare column of the `Q` buffer is where a residual direction, an augmentation column or a unit vector is built, and the spare row of the `R` buffer is where a Hessenberg fill or a new row of `A` lives while a verb runs. Rotations are `LinearAlgebra.Givens` applied with stdlib `lmul!`/`rmul!`, never a hand-written kernel. Three verbs disturb `R` and restore it through one shared retriangularization function that skips already-zero entries, so a band-limited disturbance costs only its band. Every verb validates before it mutates, so `issuccess(F)` is a constant `true` and there is no `info` field.

**Tech Stack:** Julia 1.12, `LinearAlgebra`, `TypeContracts` 0.14, `StrictModeTest` 0.4.2 (test only), `ForwardDiff`, `OffsetArrays`, `TestItems`/`TestItemRunner`, DocumenterVitepress.

**Spec:** `docs/superpowers/specs/2026-09-08-updatablefactorizations-design.md`, section 12 milestone M2b; requirements R5, R7 (QR half), R9, R10, R11, R15.

**Verified design.** The architecture below was prototyped and swept in a warm Julia 1.12.7 session before this plan was written: all six verbs over `T ∈ (Float64, ComplexF64)` and shapes `(10,5) (8,7) (6,6) (9,1) (20,7) (30,12) (9,5)` with every index swept, reaching `norm(Q*R − target)/norm(A) ≤ 1.03e-15`, `norm(Q'Q − I) ≤ 2.36e-15`, and every subdiagonal entry of the *stored* `n x n` block exactly `0.0`; the full `delete_row!` measured 0 bytes at `m,n = 200,60` for both element types. Do not "improve" a kernel without re-running the same sweep. Five facts that the kernels depend on, each measured rather than assumed:

| fact | consequence |
| --- | --- |
| `qr(::Matrix{BigFloat})` returns `QR`, not `QRCompactWY` | the constructor takes `Union{QR, QRCompactWY}` |
| `adjoint(::Givens)` is `Givens(i1, i2, conj(c), -s)` with `c` real | no hand-written adjoint-rotation helper |
| the entry a rotation "zeroes" measures 5.6e-17, not exactly zero | every kernel writes the zero explicitly |
| `sqrt(1 - norm(Q'e_i)^2)` reaches exactly `0.0` at a true `gamma` of 2.6e-15 and goes negative at 2.6e-9 | `delete_row!` uses the norm of the computed residual, never the algebraic form |
| `norm(tril(UpperTriangular(M), -1)) == 0` is `true` for any `M`, junk below the diagonal included | triangularity is asserted on the stored block, never through `F.R` |

## Global Constraints

- License MIT. No GPL code in the dependency graph outside `bench/`. `QRupdatesFast` is GPL-linked and may appear only in `bench/Project.toml`.
- Never read `qrupdate-ng` source. It is a benchmark target and routine-list reference only.
- `julia = "1.12"`. `TypeContracts = "0.14"`.
- Every algorithm's docstring cites its source article (spec section 6), and `docs/src/provenance.md` gains the same citation in the same commit as the code.
- No `@inbounds` anywhere in this plan.
- Comments state what is true of the code now. No references to this plan, to milestones, to tasks, or to how the code came to be.
- American spellings.
- The package is measured slower than LAPACK. No docstring, doc page or commit message may claim otherwise; the unconditional reorthogonalization pass makes the QR verbs slower still, and that is stated rather than smoothed over.
- Run tests with the JuliaMCP `julia_run_testitems` tool, always passing `max_workers`. Do not shell out to `Pkg.test()` except as the final pre-commit gate.
- `Project.toml` and `test/Project.toml` are edited only through `Pkg` from an MCP Julia session, never by hand-written UUID. This milestone adds no dependency, so no edit is expected.
- Every verb either completes or leaves the factorization exactly as it was: same size, same capacity, same factors, same zeroed workspace. Two verbs cannot decide before they compute — `insert_column!` and `delete_row!` build a candidate direction in the augmentation column and measure it — so each clears that column on its throw path. Nothing else is touched before the guard. `LinearAlgebra.issuccess(::UpdatableQR)` is a constant `true` and stays one.

## File Structure

| file | responsibility |
| --- | --- |
| `src/UpdatableFactorizations.jl` | three new includes, four new exports, four `public` names |
| `src/qr_rep.jl` | `AbstractQRep`, its `@contract`, `DenseQ` and its storage helpers |
| `src/qr_type.jl` | `UpdatableQR`, constructors, `qr_householder`, accessors, `Factorization` interface, `_grow!` |
| `src/qr_update.jl` | `_project!`, `_retriangularize!`, and the six verbs |
| `src/contracts.jl` | `@invariants UpdatableQR` appended |
| `test/qr_rep.jl` | contract conformance and `DenseQ` storage behavior |
| `test/qr_type.jl` | construction, properties, solving, capacity growth |
| `test/qr_update.jl` | the six verbs, their index sweeps and their failure paths |
| `test/generic.jl` | element types, seeded partials and offset inputs extended to the QR verbs |
| `test/strict.jl` | invariants chain, allocation gate and type-stability gate extended |
| `docs/src/q_representations.md` | the `AbstractQRep` seam and `DenseQ` |
| `docs/src/updating.md`, `docs/src/provenance.md` | QR verbs, new citations |

`docs/src/api.md` is not edited: it is an `@autodocs` block over the whole module, which picks up
every newly documented name, exported or `public`, on its own. No page adds an explicit `@docs`
block for a name that block already covers; Documenter reports the second copy as a duplicate
docstring.

## Workspace map

Published before the code, and binding: buffer aliasing is a correctness bug that no allocation test catches. Every buffer holds one role at a time.

| buffer | extent | written by | live range |
| --- | --- | --- | --- |
| `qrep.buf[1:m, n+1]` (spare column) | `m` | `lowrankupdate!` (normalized residual), `insert_column!` (same), `delete_row!` (augmentation column), `insert_row!` (`e_i`) | one verb; zero on entry and on exit, including the exit through a throw |
| `factors[n+1, 1:n]` (spare row) | `n` | `lowrankupdate!` (Hessenberg fill), `insert_row!` (the new row), `delete_row!` (ends holding the deleted row of `A`) | one verb; zero on entry and on exit |
| `factors[1:n, n+1]` (spare column) | `n` | `shift_columns!` (the moved column of `R`) | one verb; zero on entry and on exit |
| `work[1:(n+1)]` | `ncap+1` | `lowrankupdate!` (`z = [Q'u; rho]`), `insert_column!` (`Q'x`), `delete_row!` (`Q'e_i`), `ldiv!(F, B)` | one verb |
| `corr[1:n]` | `ncap+1` | the reorthogonalization correction in `lowrankupdate!`, `insert_column!`, `delete_row!` | one verb |

`delete_column!` uses no scratch vector at all. `insert_column!` delegates to `shift_columns!`, and the two do not collide because `shift_columns!` holds the moved column of `R` in the `R` buffer's spare column rather than in `work`.

---

### Task 1: Rule on the two `rtol` defaults and on `StrictMode`

**Files:**
- Modify: `.superpowers/sdd/2026-09-08-m0-m2a-cholesky-lu-updating/progress.md` (or this milestone's own ledger, if one is opened)

**Interfaces:**
- Consumes: nothing
- Produces: three recorded rulings that Tasks 8, 10 and 11 implement

These are the decisions the design cannot settle on its own. No code that depends on one of them is written before it is ruled: `delete_row!` and `insert_column!` wait on the first two, and the Task 11 gates wait on the third.

#### Ruling 1: `delete_row!`'s default `rtol`

Measured: with the reorthogonalization pass, `gamma` at *exact* leverage one comes out **4.6e-17**, not zero — the pass manufactures a unit direction out of rounding noise. So `rtol = 0` makes the spec's named failure path ("when `gamma` is at or near zero … `delete_row!` throws") essentially unreachable, and the verb silently returns a factorization whose `norm(Q'Q - I)` is 8.9e-16 for a matrix whose numerical rank has dropped.

| option | default | consequence |
| --- | --- | --- |
| A | `rtol = 0` | consistent with the shipped `lowrankupdate!(::UpdatableLU, u, v; rtol = 0)`; the documented throw is effectively dead code and the docstring must say a caller has to pass `rtol` to reach it |
| B (recommended) | `rtol = sqrt(eps(real(T)))` | `gamma` is dimensionless and lies in `[0,1]`, unlike an LU pivot, so a nonzero default is well defined here in a way it was not there; measured, `gamma`, the smallest diagonal of the resulting `R`, and `sigma_min(A[keep, :])` agree to five digits across eight decades, so `rtol` is a direct threshold on how close the deletion comes to dropping the rank |

#### Ruling 2: `insert_column!`'s default `rtol`

The same measurement drives this one, so it is ruled in the same breath. `insert_column!`'s guard is `rho > rtol * xnrm`, so `rtol = 0` reduces it to `rho > 0`. Measured at `m,n = 9,4` with `x` an exact copy of an existing column and the two projection passes the implementation performs: `rho = 1.95e-16`, strictly positive, `rho / xnrm = 5.6e-17`. The column is admitted, and the diagonal entry a later solve divides by is rounding noise. Unlike `rho`, the ratio `rho / xnrm` is dimensionless and lies in `[0,1]`, exactly as `gamma` does, so a relative default is as well defined here as there.

| option | default | consequence |
| --- | --- | --- |
| A | `rtol = 0` | the documented throw is unreachable for an exactly dependent column; the docstring must say a caller has to pass `rtol` to reach it, and the dependence test in `test/qr_update.jl` passes an explicit `rtol` |
| B (recommended) | `rtol = sqrt(eps(real(T)))` | measured, this rejects an exact duplicate column (ratio 5.6e-17) and admits a generic one (ratio 0.72); the threshold is on `rho / norm(x)`, which the docstring states |

The two rulings should agree. A package where one verb defaults to a relative threshold and the sibling defaults to zero is harder to document than either choice on its own.

#### Ruling 3: `StrictMode` in `[deps]`

R10 reads "StrictMode assertions on `DenseQ` rank-1 kernels, StrictModeTest gating CI", and spec section 8 lists `StrictMode` in `Project.toml` `[deps]`. The shipped `Project.toml` carries `LinearAlgebra` and `TypeContracts` only, and the M0/M2a ledger records no ruling that drops `StrictMode`, so there is no precedent to follow and R10's first half is unmet.

| option | consequence |
| --- | --- |
| A | Add `StrictMode` to `[deps]` and place `@assert_noalloc`/`@assert_typestable` inside the `DenseQ` rank-1 kernels, as spec section 8 and R10 both say. This is a runtime dependency of the package, not of its tests. |
| B | Ship the `StrictModeTest`-only gate and amend R10 in the spec to match, so the requirement and the code agree. |

Do not carry R10 as satisfied under Option B until the spec row is amended.

- [ ] **Step 1: Put all three rulings to the user**

Quote the tables above. Each is a difference in what the package promises; none is an implementer's call.

- [ ] **Step 2: Record the rulings in the decision ledger**

Add them next to the `uplo` and `pivoted` divergences, each stating the choice and the measurement or requirement text that justified it.

- [ ] **Step 3: Note where each ruling lands**

The plan below is written for **Option B on both `rtol` rulings**. If Option A is ruled for `delete_row!`, exactly two edits change:

1. In `src/qr_update.jl`, `delete_row!`'s keyword becomes `rtol::Real = 0`, and its docstring's paragraph on the default is rewritten to say a caller must pass `rtol` for the leverage test to fire.
2. In `test/qr_update.jl`, the leverage-sweep item's `@test 0 < thrown < 8` becomes `@test thrown == 0`, and the paired `rtol` item's "succeeds with the default" branch keeps `rtol = 0` explicit.

If Option A is ruled for `insert_column!`, exactly two edits change:

1. In `src/qr_update.jl`, `insert_column!`'s keyword becomes `rtol::Real = 0`, and its docstring says the default admits any column whose residual is nonzero, so a caller must pass `rtol` to reach the throw.
2. In `test/qr_update.jl`, the dependence item passes `rtol = 1.0e-8` explicitly instead of relying on the default, and the paired `rtol` item drops its "the default admits it" branch.

Ruling 3 lands in `Project.toml`, in the `DenseQ` rank-1 kernels, and in the Self-review's R10 line.

---

### Task 2: `AbstractQRep`, its contract, and `DenseQ`

**Files:**
- Create: `src/qr_rep.jl`, `test/qr_rep.jl`
- Modify: `src/UpdatableFactorizations.jl`

**Interfaces:**
- Consumes: `LinearAlgebra.Givens`, `TypeContracts.@contract`, `TypeContracts.@verify`
- Produces: `AbstractQRep{T}`; `materialize`; `DenseQ{T,S}`; `_active`, `_augmented`, `_spare`, `_insertrow!`, `_deleterow!`, `_dropcolumn!`, `_clearspare!`; `capacity(::DenseQ)`

- [ ] **Step 1: Write the failing test**

`test/qr_rep.jl`:

```julia
@testitem "DenseQ implements the AbstractQRep contract" begin
    using LinearAlgebra
    using UpdatableFactorizations: AbstractQRep, DenseQ, materialize
    using UpdatableFactorizations.TypeContracts: @test_implements
    @test_implements DenseQ AbstractQRep
end

@testitem "DenseQ presents its active block and keeps the rest zero" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: DenseQ, materialize, _active, _spare

    Random.seed!(20260908)
    m, n = 6, 3
    buf = zeros(6, 5)
    A = qr(randn(m, n)).Q * Matrix(I, m, n)
    copyto!(view(buf, 1:m, 1:n), A)
    q = DenseQ{Float64, Matrix{Float64}}(buf, m, n)

    @test size(q) == (m, n)
    @test size(q, 1) == m
    @test size(q, 3) == 1
    @test_throws "dimension must be positive" size(q, 0)
    @test eltype(q) === Float64
    @test Matrix(q) == A
    @test materialize(q) === q
    @test all(iszero, _spare(q))
end

@testitem "DenseQ applies rotations to the augmented factor" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: DenseQ, _active, _augmented

    Random.seed!(20260908)
    for T in (Float64, ComplexF64)
        m, n = 7, 4
        buf = zeros(T, m, n + 1)
        copyto!(view(buf, 1:m, 1:n), qr(randn(T, m, n)).Q * Matrix{T}(I, m, n))
        q = DenseQ{T, Matrix{T}}(buf, m, n)
        c, s, _ = LinearAlgebra.givensAlgorithm(one(T), one(T))
        G = LinearAlgebra.Givens(2, 3, T(c), T(s))

        # Orthonormality alone is preserved by doing nothing, so each side is checked against
        # the rotation written out as a matrix. `Matrix(::Givens, n)` does not exist; rotating
        # an identity of the right size is how the matrix is formed.
        Gright = rmul!(Matrix{T}(I, n + 1, n + 1), G)
        Gleft = lmul!(G, Matrix{T}(I, m, m))

        ref = copy(_augmented(q))
        rmul!(q, G)
        @test _augmented(q) ≈ ref * Gright
        @test norm(_active(q)' * _active(q) - I) < 1.0e-14

        ref = copy(_augmented(q))
        lmul!(G, q)
        @test _augmented(q) ≈ Gleft * ref
        @test norm(_active(q)' * _active(q) - I) < 1.0e-14
    end
end

@testitem "DenseQ structural edits leave storage outside the active block zero" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: DenseQ, _insertrow!, _deleterow!, _dropcolumn!, _spare

    Random.seed!(20260908)
    m, n = 5, 3
    buf = zeros(8, n + 1)
    B = randn(m, n)
    copyto!(view(buf, 1:m, 1:n), B)
    q = DenseQ{Float64, Matrix{Float64}}(buf, m, n)

    _insertrow!(q, 2)
    @test size(q) == (m + 1, n)
    @test view(q.buf, 2, 1:n) == zeros(n)
    @test view(q.buf, 3:(m + 1), 1:n) == B[2:m, :]

    _deleterow!(q, 2)
    @test size(q) == (m, n)
    @test view(q.buf, 1:m, 1:n) == B
    @test all(iszero, view(q.buf, (m + 1):8, :))
    @test all(iszero, _spare(q))

    _dropcolumn!(q)
    @test size(q) == (m, n - 1)
    @test all(iszero, view(q.buf, 1:m, n:(n + 1)))
end
```

- [ ] **Step 2: Run the tests to verify they fail**

`julia_run_testitems` with `max_workers = 4`, filtered to the four items above.
Expected: FAIL — `AbstractQRep` and `DenseQ` are not defined.

- [ ] **Step 3: Write the implementation**

`src/qr_rep.jl`:

```julia
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
    Base.size(::Self)::Tuple{Int, Int} => "the shape of the active factor"
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
mutable struct DenseQ{T, S <: AbstractMatrix{T}} <: AbstractQRep{T}
    buf::S
    m::Int
    n::Int
end

# The active factor, the factor plus its augmentation column, and that column alone. All three
# are one concrete SubArray type, so every kernel has a single specialization.
_active(q::DenseQ) = view(q.buf, 1:q.m, 1:q.n)
_augmented(q::DenseQ) = view(q.buf, 1:q.m, 1:(q.n + 1))
_spare(q::DenseQ) = view(q.buf, 1:q.m, q.n + 1)

Base.size(q::DenseQ) = (q.m, q.n)
Base.copyto!(A::AbstractMatrix, q::DenseQ) = copyto!(A, _active(q))
materialize(q::DenseQ) = q

# The largest shape reachable before the buffer is reallocated. The trailing column is the
# augmentation every verb works in, not capacity a caller may fill.
capacity(q::DenseQ) = (size(q.buf, 1), size(q.buf, 2) - 1)

LinearAlgebra.lmul!(G::Givens, q::DenseQ) = (lmul!(G, _augmented(q)); q)
LinearAlgebra.rmul!(q::DenseQ, G::Givens) = (rmul!(_augmented(q), G); q)

# Structural changes. Each leaves storage outside the active block zero.
function _insertrow!(q::DenseQ{T}, i::Integer) where {T}
    for j in 1:q.n, r in q.m:-1:i
        q.buf[r + 1, j] = q.buf[r, j]
    end
    for j in 1:q.n
        q.buf[i, j] = zero(T)
    end
    q.m += 1
    return q
end

function _deleterow!(q::DenseQ{T}, i::Integer) where {T}
    for j in 1:q.n, r in i:(q.m - 1)
        q.buf[r, j] = q.buf[r + 1, j]
    end
    for j in 1:(q.n + 1)
        q.buf[q.m, j] = zero(T)
    end
    for r in 1:q.m
        q.buf[r, q.n + 1] = zero(T)
    end
    q.m -= 1
    return q
end

_dropcolumn!(q::DenseQ{T}) where {T} = (fill!(view(q.buf, 1:q.m, q.n), zero(T)); q.n -= 1; q)

_clearspare!(q::DenseQ{T}) where {T} = (fill!(_spare(q), zero(T)); q)

@verify DenseQ
```

The four structural helpers are `DenseQ`-specific and stay out of the contract; the contract is widened when a second implementation exists to widen it against.

`Base.Matrix` cannot be a contract clause: TypeContracts 0.14 raises `MethodError: Cannot convert an object of type Type{Matrix} to an object of type Function`, because a constructor is not a `Function`. `copyto!` is the operation both `Matrix` and the verbs go through, so the clause count, the operation set and the spec's five named operations are all preserved.

- [ ] **Step 4: Wire the include and the public names**

`src/UpdatableFactorizations.jl` — add `Givens` to the `LinearAlgebra` import list, add the include, and declare the names:

```julia
using LinearAlgebra
using LinearAlgebra: givensAlgorithm, Givens, PosDefException, ZeroPivotException
import LinearAlgebra: lowrankupdate!, lowrankdowndate!, ldiv!, logdet, det

export UpdatableCholesky, UpdatableLU, UpdatableQR
export insert_column!, delete_column!, shift_columns!, insert_row!, delete_row!
export qr_householder
public AbstractQRep, DenseQ, materialize, capacity

include("cholesky_type.jl")
include("cholesky_update.jl")
include("cholesky_resize.jl")
include("lu_type.jl")
include("lu_update.jl")
include("qr_rep.jl")
include("qr_type.jl")
include("qr_update.jl")
include("contracts.jl")
```

`src/qr_rep.jl` uses `@contract` and `@verify`, which live in `TypeContracts`. `src/contracts.jl` currently carries the only `using TypeContracts`; move that line into the module file above the includes so both files see it, and delete it from `src/contracts.jl`.

Create `src/qr_type.jl` and `src/qr_update.jl` for now with nothing but the bare function declarations the exports resolve against. `src/qr_type.jl`:

```julia
function qr_householder end
```

`src/qr_update.jl`:

```julia
function insert_row! end
function delete_row! end
```

- [ ] **Step 5: Run the tests to verify they pass**

Expected: PASS, four items.

- [ ] **Step 6: Commit**

```bash
git add src/qr_rep.jl src/qr_type.jl src/qr_update.jl src/UpdatableFactorizations.jl src/contracts.jl test/qr_rep.jl
git commit -m "Add the orthonormal-factor representation seam and DenseQ"
```

---

### Task 3: `UpdatableQR` type, constructors and capacity growth

**Files:**
- Create: `test/qr_type.jl`
- Modify: `src/qr_type.jl`

**Interfaces:**
- Consumes: `AbstractQRep`, `DenseQ`, `_active`, `capacity(::DenseQ)`
- Produces: `UpdatableQR{T,S,Q}`; `UpdatableQR(::Union{QR,QRCompactWY}; capacity)`; `UpdatableQR(::AbstractMatrix; capacity)`; `UpdatableQR(Q::AbstractMatrix, R::AbstractMatrix; capacity)`; `qr_householder`; `_upper`, `_raug`, `_rspare`; `capacity(::UpdatableQR)`; `size`, `getproperty`, `propertynames`, `AbstractMatrix`, `Matrix`, `issuccess`; `_grow!(F, mneeded, nneeded)`

- [ ] **Step 1: Write the failing test**

`test/qr_type.jl`:

```julia
@testitem "UpdatableQR reconstructs the factored matrix" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((9, 5), (6, 6), (8, 7), (9, 1))
        A = randn(T, m, n)
        F = UpdatableQR(A)
        @test size(F) == (m, n)
        @test size(F, 1) == m
        @test size(F, 2) == n
        @test size(F, 3) == 1
        @test norm(F.Q * F.R - A) / norm(A) < 1.0e-13
        @test norm(F.Q' * F.Q - I) < 1.0e-13
        # `F.R` wraps the stored block in `UpperTriangular`, which reports a zero subdiagonal
        # whatever the block holds, so triangularity is asserted on the block itself. Every
        # other item in this milestone does the same, for the same reason.
        Rs = getfield(F, :factors)
        @test all(iszero, [Rs[i, j] for j in 1:F.n for i in (j + 1):F.n])
        @test issuccess(F)
        @test norm(Matrix(F) - A) / norm(A) < 1.0e-13
    end
end

@testitem "UpdatableQR zeroes the reflector storage the standard library leaves behind" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    m, n = 9, 5
    A = randn(m, n)
    G = qr(A)
    # The positive control: LinearAlgebra stores the reflectors below the diagonal of `factors`.
    # Without it this item goes hollow if a future release returns a clean triangle.
    @test any(!iszero, [getfield(G, :factors)[i, j] for j in 1:n for i in (j + 1):m])
    F = UpdatableQR(G)
    R = getfield(F, :factors)
    @test all(iszero, [R[i, j] for j in 1:n for i in (j + 1):size(R, 1)])
    @test all(iszero, view(R, :, (n + 1):size(R, 2)))
    Q = getfield(F, :qrep)
    @test all(iszero, view(Q.buf, (m + 1):size(Q.buf, 1), :))
    @test all(iszero, view(Q.buf, :, (n + 1):size(Q.buf, 2)))
end

@testitem "UpdatableQR exposes live views of its factors" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity

    Random.seed!(20260908)
    F = UpdatableQR(randn(7, 4))
    @test propertynames(F) == (:Q, :R)
    @test :qrep in propertynames(F, true)
    @test :factors in propertynames(F, true)
    Q = F.Q
    Q[1, 1] += 1.0
    @test F.Q[1, 1] == Q[1, 1]        # a view of live storage, not a copy
    @test capacity(F) == (14, 8)
    @test capacity(UpdatableQR(randn(7, 4); capacity = (7, 4))) == (7, 4)
end

@testitem "UpdatableQR rejects inconsistent construction arguments" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    A = randn(6, 3)
    @test_throws "row capacity 5 is below the size 6" UpdatableQR(A; capacity = (5, 3))
    @test_throws "column capacity 2 is below the size 3" UpdatableQR(A; capacity = (6, 2))
    @test_throws "it requires m >= n" UpdatableQR(qr(randn(3, 6)))
    @test_throws "dimension must be positive" size(UpdatableQR(A), 0)
    # The guard is an exact-zero test on the diagonal, so the fixture makes a diagonal entry
    # exactly zero. Householder QR of a matrix that is only numerically rank deficient does
    # not: for `B[:, 3] = B[:, 1]` the third diagonal entry measures 1.3e-16, and the
    # factorization is accepted. `insert_column!`'s `rtol` is the relative test.
    C = randn(6, 3)
    C[:, 2] .= 0.0
    @test_throws "is rank deficient" UpdatableQR(C)
end

@testitem "UpdatableQR constructs from a non-BLAS factorization" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    A = BigFloat.(randn(7, 4))
    G = qr(A)
    @test G isa LinearAlgebra.QR          # not QRCompactWY; the constructor takes both
    F = UpdatableQR(G)
    @test norm(F.Q * F.R - A) / norm(A) < 1.0e-30
end

@testitem "UpdatableQR wraps an already-computed thin factorization" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((9, 5), (6, 6))
        A = randn(T, m, n)
        G = qr(A)
        F = UpdatableQR(Matrix(G.Q), Matrix(G.R); capacity = (m, n))
        @test capacity(F) == (m, n)
        @test norm(Matrix(F) - A) / norm(A) < 1.0e-13
        @test norm(F.Q' * F.Q - I) < 1.0e-13
    end
    @test_throws "Q is 6 by 4 and R is 3 by 3" UpdatableQR(randn(6, 4), randn(3, 3))
    @test_throws "R is 4 by 3; it must be square" UpdatableQR(randn(6, 4), randn(4, 3))
end

@testitem "qr_householder factors through the standard library" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((9, 5), (6, 6), (9, 1))
        A = randn(T, m, n)
        F = qr_householder(A)
        @test F isa UpdatableQR
        @test norm(Matrix(F) - A) / norm(A) < 1.0e-13
        # The accuracy claim the docs make for this path: orthogonality at machine precision,
        # unconditionally, because the factorization is LinearAlgebra's.
        @test norm(F.Q' * F.Q - I) < 10 * eps(real(T)) * n
    end
    @test capacity(qr_householder(randn(9, 5); capacity = (9, 5))) == (9, 5)
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — `UpdatableQR` not defined.

- [ ] **Step 3: Write the implementation**

`src/qr_type.jl`:

```julia
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
```

Once `getproperty` is overridden, internal code reads `qrep` through `getfield`; `m`, `n`, `factors`, `work` and `corr` are not intercepted and use dot syntax, as `UpdatableCholesky` does.

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS, seven items.

- [ ] **Step 5: Commit**

```bash
git add src/qr_type.jl test/qr_type.jl
git commit -m "Add the UpdatableQR type, its constructors and its capacity growth"
```

---

### Task 4: Solving

**Files:**
- Modify: `src/qr_type.jl`, `test/qr_type.jl`

**Interfaces:**
- Consumes: `UpdatableQR`, `_upper`, `F.Q`
- Produces: `LinearAlgebra.ldiv!(y, F, b)`, `LinearAlgebra.ldiv!(F, B)`, `Base.:\(F, b)`, `Base.:\(F, B)`

- [ ] **Step 1: Write the failing test**

Append to `test/qr_type.jl`:

```julia
@testitem "UpdatableQR solves least squares problems" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((9, 5), (6, 6), (9, 1))
        A = randn(T, m, n)
        b = randn(T, m)
        F = UpdatableQR(A)
        x = F \ b
        # A least-squares solution has n entries; the generic Factorization fallback returns m.
        @test length(x) == n
        @test norm(A' * (A * x - b)) / norm(A' * b) < 1.0e-11

        y = similar(b, n)
        ldiv!(y, F, b)
        @test y ≈ x

        B = randn(T, m, 3)
        X = F \ B
        @test size(X) == (n, 3)
        @test norm(A' * (A * X - B)) / norm(A' * B) < 1.0e-11

        C = copy(B)
        ldiv!(F, C)
        @test C[1:n, :] ≈ X
    end
end

@testitem "UpdatableQR solving rejects mismatched shapes and offset arguments" begin
    using LinearAlgebra, OffsetArrays, Random

    Random.seed!(20260908)
    F = UpdatableQR(randn(9, 5))
    @test_throws "b has length 4, factorization is 9x5" ldiv!(zeros(5), F, zeros(4))
    @test_throws "y has length 4, factorization is 9x5" ldiv!(zeros(4), F, zeros(9))
    @test_throws "B has 4 rows, factorization is 9x5" ldiv!(F, zeros(4, 2))
    # The solve path is one-based by declaration, unlike the updating verbs.
    @test_throws "offset arrays are not supported" ldiv!(
        zeros(5), F, OffsetVector(zeros(9), 0:8)
    )
    @test_throws "offset arrays are not supported" F \ OffsetVector(zeros(9), 0:8)
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — `F \ b` returns a length-`m` vector of the wrong content, and the three-argument `ldiv!` hits the generic fallback.

- [ ] **Step 3: Write the implementation**

Append to `src/qr_type.jl`:

```julia
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

# The least-squares solution has n entries; the generic Factorization fallback would return m.
Base.:\(F::UpdatableQR, b::AbstractVector) =
    ldiv!(similar(b, eltype(F), Base.OneTo(size(F, 2))), F, b)
Base.:\(F::UpdatableQR, B::AbstractMatrix) =
    ldiv!(F, LinearAlgebra.copy_similar(B, eltype(F)))[1:size(F, 2), :]
```

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qr_type.jl test/qr_type.jl
git commit -m "Add least-squares solving for UpdatableQR"
```

---

### Task 5: Retriangularization and column deletion

**Files:**
- Modify: `src/qr_update.jl`
- Create: `test/qr_update.jl`

**Interfaces:**
- Consumes: `UpdatableQR`, `DenseQ`, `_dropcolumn!`, `givensAlgorithm`, `Givens`
- Produces: `_retriangularize!(Rv, q)`; `delete_column!(F::UpdatableQR, j)`

- [ ] **Step 1: Write the failing test**

`test/qr_update.jl`:

```julia
@testitem "QR column deletion, every index" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (7, 7), (9, 1))
        A = randn(T, m, n)
        # The fixture must not already be near-orthogonal, or a broken rotation sweep passes.
        @test norm(A' * A - I) > 1
        for j in 1:n
            F = UpdatableQR(A)
            delete_column!(F, j)
            keep = setdiff(1:n, j)
            @test size(F) == (m, n - 1)
            if n > 1
                @test norm(F.Q * F.R - A[:, keep]) / norm(A) < 1.0e-12
                @test norm(F.Q' * F.Q - I) < 1.0e-12
            end
            Q = getfield(F, :qrep)
            R = getfield(F, :factors)
            @test all(iszero, [R[i, j] for j in 1:F.n for i in (j + 1):F.n])
            @test all(iszero, view(Q.buf, :, (F.n + 1):size(Q.buf, 2)))
            @test all(iszero, view(R, (F.n + 1):size(R, 1), :))
            @test all(iszero, view(R, :, (F.n + 1):size(R, 2)))
        end
    end
end

@testitem "QR column deletion rejects an out-of-range index" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    F = UpdatableQR(randn(8, 4))
    @test_throws BoundsError delete_column!(F, 5)
    @test_throws BoundsError delete_column!(F, 0)
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — no `delete_column!` method for `UpdatableQR`.

- [ ] **Step 3: Write the implementation**

Append to `src/qr_update.jl`:

```julia
# Zero every subdiagonal entry of `Rv` with row rotations, mirrored onto `q`'s columns so that
# Q*R is unchanged. Entries that are already zero are skipped, so a disturbance confined to a
# band costs only that band. Column-major order with the rows taken from the bottom up covers
# both shapes that arise: a Hessenberg subdiagonal, and a single spike in one column.
function _retriangularize!(Rv::AbstractMatrix{T}, q::AbstractQRep{T}) where {T}
    nr, nc = size(Rv)
    for c in 1:nc, r in min(nr, nc + 1):-1:(c + 1)
        iszero(Rv[r, c]) && continue
        cc, ss, rr = givensAlgorithm(Rv[r - 1, c], Rv[r, c])
        G = Givens(r - 1, r, oftype(Rv[r, c], cc), oftype(Rv[r, c], ss))
        lmul!(G, Rv)
        rmul!(q, G')
        Rv[r - 1, c] = rr
        Rv[r, c] = zero(T)    # lmul! leaves a rounding residue; the invariant is an exact zero
    end
    return Rv
end

"""
    delete_column!(F::UpdatableQR, j) -> F

Remove column `j` of the factored matrix, in `O((n - j)(m + n))` operations.

Sliding the trailing columns of `R` one place to the left leaves it upper Hessenberg from column
`j` on, and rotations chase that subdiagonal back to zero. The columns before `j` are untouched,
so an early deletion costs the whole factor and a late one costs almost nothing.

Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5.
"""
function delete_column!(F::UpdatableQR{T, S, <:DenseQ}, j::Integer) where {T, S}
    n = F.n
    1 <= j <= n || throw(BoundsError(F, j))
    q = getfield(F, :qrep)
    R = getfield(F, :factors)
    for c in j:(n - 1), r in 1:n
        R[r, c] = R[r, c + 1]
    end
    for r in 1:n
        R[r, n] = zero(T)
    end
    n == 1 || _retriangularize!(view(R, 1:n, 1:(n - 1)), q)
    _dropcolumn!(q)
    F.n = n - 1
    return F
end
```

The rotations `_retriangularize!` forms here act in planes `(r-1, r)` with `r <= n`, so they never reach the augmentation column and it stays zero.

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qr_update.jl test/qr_update.jl
git commit -m "Add QR column deletion"
```

---

### Task 6: Column shifting

**Files:**
- Modify: `src/qr_update.jl`, `test/qr_update.jl`

**Interfaces:**
- Consumes: `_retriangularize!`, `_rspare`
- Produces: `shift_columns!(F::UpdatableQR, i, j)`

- [ ] **Step 1: Write the failing test**

Append to `test/qr_update.jl`:

```julia
@testitem "QR column shifting, every index pair" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (7, 7))
        A = randn(T, m, n)
        @test norm(A' * A - I) > 1
        for i in 1:n, j in 1:n
            F = UpdatableQR(A)
            shift_columns!(F, i, j)
            p = collect(1:n)
            deleteat!(p, i)
            insert!(p, j, i)
            @test norm(F.Q * F.R - A[:, p]) / norm(A) < 1.0e-12
            @test norm(F.Q' * F.Q - I) < 1.0e-12
            R = getfield(F, :factors)
            @test all(iszero, [R[i, j] for j in 1:F.n for i in (j + 1):F.n])
            @test all(iszero, view(R, :, (n + 1):size(R, 2)))
        end
    end
end

@testitem "QR column shifting rejects out-of-range indices" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    F = UpdatableQR(randn(8, 4))
    @test_throws BoundsError shift_columns!(F, 5, 1)
    @test_throws BoundsError shift_columns!(F, 1, 0)
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — no `shift_columns!` method for `UpdatableQR`.

- [ ] **Step 3: Write the implementation**

Append to `src/qr_update.jl`:

```julia
"""
    shift_columns!(F::UpdatableQR, i, j) -> F

Move column `i` of the factored matrix to position `j`, sliding the columns between them by one,
in `O(|i - j|(m + n))` operations.

Moving a column right leaves `R` upper Hessenberg over the columns it passed; moving it left
leaves a single spike in column `j`. Rotations clear both.

Reichel and Gragg, *Algorithm 686: FORTRAN subroutines for updating the QR decomposition*,
ACM Transactions on Mathematical Software 16 (1990), 369-377.
"""
function shift_columns!(F::UpdatableQR{T, S, <:DenseQ}, i::Integer, j::Integer) where {T, S}
    n = F.n
    1 <= i <= n || throw(BoundsError(F, i))
    1 <= j <= n || throw(BoundsError(F, j))
    i == j && return F
    q = getfield(F, :qrep)
    R = getfield(F, :factors)
    hold = _rspare(F)
    for r in 1:n
        hold[r] = R[r, i]
    end
    if i < j
        for c in i:(j - 1), r in 1:n
            R[r, c] = R[r, c + 1]
        end
    else
        for c in i:-1:(j + 1), r in 1:n
            R[r, c] = R[r, c - 1]
        end
    end
    for r in 1:n
        R[r, j] = hold[r]
        hold[r] = zero(T)
    end
    _retriangularize!(view(R, 1:n, 1:n), q)
    return F
end
```

The moved column lives in the `R` buffer's spare column rather than in `work`, so `insert_column!` can delegate here while holding its own scratch.

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS. The two shapes and both element types together sweep 148 index pairs.

- [ ] **Step 5: Commit**

```bash
git add src/qr_update.jl test/qr_update.jl
git commit -m "Add QR column shifting"
```

---

### Task 7: Rank-1 update

**Files:**
- Modify: `src/qr_update.jl`, `test/qr_update.jl`

**Interfaces:**
- Consumes: `_active`, `_spare`, `_augmented`, `_clearspare!`, `_raug`, `_retriangularize!`
- Produces: `_project!(w, r, Qa, corr)`, `_project_residual!(w, r, Qa, corr)`; `LinearAlgebra.lowrankupdate!(F::UpdatableQR, u, v)`

- [ ] **Step 1: Write the failing test**

Append to `test/qr_update.jl`:

```julia
@testitem "QR rank-1 update" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (8, 7), (6, 6), (9, 1), (30, 12))
        A = randn(T, m, n)
        u = randn(T, m)
        v = randn(T, n)
        F = UpdatableQR(A)
        Qb = copy(F.Q)
        lowrankupdate!(F, u, v)
        @test norm(F.Q * F.R - (A + u * v')) / norm(A) < 1.0e-12
        @test norm(F.Q' * F.Q - I) < 1.0e-12
        R = getfield(F, :factors)
        @test all(iszero, [R[i, j] for j in 1:F.n for i in (j + 1):F.n])
        # A generic `u` has a residual outside the range of `Q`, so the update takes the branch
        # that admits a new direction and the range moves. Measured, this quantity is 0.92 here
        # and 1.5e-15 when the other branch runs. It is the control for the in-range item below.
        if m > n
            @test norm(F.Q - Qb * (Qb' * F.Q)) > 1.0e-6
        end
    end
end

@testitem "QR rank-1 update with an in-range vector" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (6, 6))
        A = randn(T, m, n)
        F = UpdatableQR(A)
        y = randn(T, n)
        u = F.Q * y
        v = randn(T, n)
        Qb = copy(F.Q)
        lowrankupdate!(F, u, v)
        @test norm(F.Q * F.R - (A + u * v')) / norm(A) < 1.0e-12
        @test norm(F.Q' * F.Q - I) < 1.0e-12
        R = getfield(F, :factors)
        @test all(iszero, [R[i, j] for j in 1:F.n for i in (j + 1):F.n])
        # `u` lies in the range of `Q`, so the update stays inside that range: this is what the
        # branch controls, and the reconstruction residual is correct either way. On a square
        # factorization it is the only branch there is.
        @test norm(F.Q - Qb * (Qb' * F.Q)) < 1.0e-12
    end
end

@testitem "QR rank-1 update does not consume its vectors" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    m, n = 9, 5
    F = UpdatableQR(randn(m, n))
    u = randn(m)
    v = randn(n)
    uc = copy(u)
    vc = copy(v)
    lowrankupdate!(F, u, v)
    @test u == uc
    @test v == vc
end

@testitem "QR rank-1 update rejects mismatched vector lengths" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    F = UpdatableQR(randn(9, 5))
    @test_throws "u has length 8, factorization is 9x5" lowrankupdate!(F, zeros(8), zeros(5))
    @test_throws "v has length 4, factorization is 9x5" lowrankupdate!(F, zeros(9), zeros(4))
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — no `lowrankupdate!` method for `UpdatableQR`.

- [ ] **Step 3: Write the implementation**

Append to `src/qr_update.jl`:

```julia
# Project `r` onto the columns of `Qa`, leaving the coefficients in `w` and the residual in `r`,
# and return the residual norm. The second pass is unconditional: with one pass the loss of
# orthogonality grows like eps divided by the residual norm, which is unbounded as the residual
# collapses, while two passes hold it at O(eps).
#
# Giraud, Langou and Rozlozník, *The loss of orthogonality in the Gram-Schmidt orthogonalization
# process*, Computers and Mathematics with Applications 50 (2005), 1069-1075.
function _project!(w, r, Qa, corr)
    mul!(w, Qa', r)
    mul!(r, Qa, w, -1, true)
    return _project_residual!(w, r, Qa, corr)
end

# The second pass alone, for a caller that has already applied the first and accumulated its
# coefficients in `w`.
function _project_residual!(w, r, Qa, corr)
    mul!(corr, Qa', r)
    mul!(r, Qa, corr, -1, true)
    w .+= corr
    return norm(r)
end

"""
    lowrankupdate!(F::UpdatableQR, u, v) -> F

Replace the factorization of `A` with that of `A + u*v'` in `O(mn)` operations. Neither `u` nor
`v` is modified.

`u` is split into its projection onto the range of `Q` and a residual. When the residual is
negligible relative to `u`, or when the factorization is square and so has no room for a new
direction, the update is carried out inside the existing range; that is a legitimate case rather
than a failure, and the routine never throws for it.

Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating the
Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772-795.
Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5.
"""
function LinearAlgebra.lowrankupdate!(
        F::UpdatableQR{T, S, <:DenseQ}, u::AbstractVector, v::AbstractVector
    ) where {T, S}
    m, n = F.m, F.n
    length(u) == m ||
        throw(DimensionMismatch("u has length $(length(u)), factorization is $(m)x$(n)"))
    length(v) == n ||
        throw(DimensionMismatch("v has length $(length(v)), factorization is $(m)x$(n)"))
    q = getfield(F, :qrep)
    Qa = _active(q)
    r = _spare(q)
    z = view(F.work, 1:(n + 1))
    w = view(z, 1:n)
    corr = view(F.corr, 1:n)
    iu = firstindex(u) - 1
    for i in 1:m
        r[i] = u[iu + i]
    end
    unrm = norm(r)
    rho = _project!(w, r, Qa, corr)
    RA = _raug(F)
    if m > n && rho > n * eps(real(T)) * unrm
        r ./= rho
        z[n + 1] = rho
        last = n + 1
    else
        fill!(r, zero(T))
        z[n + 1] = zero(T)
        last = n
    end
    for k in (last - 1):-1:1
        c, s, rr = givensAlgorithm(z[k], z[k + 1])
        G = Givens(k, k + 1, oftype(z[k], c), oftype(z[k], s))
        z[k] = rr
        z[k + 1] = zero(T)
        lmul!(G, RA)
        rmul!(q, G')
    end
    iv = firstindex(v) - 1
    for j in 1:n
        RA[1, j] += z[1] * conj(v[iv + j])
    end
    _retriangularize!(RA, q)
    _clearspare!(q)
    for j in 1:n
        RA[n + 1, j] = zero(T)
    end
    return F
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS, four items.

- [ ] **Step 5: Commit**

```bash
git add src/qr_update.jl test/qr_update.jl
git commit -m "Add the QR rank-1 update"
```

---

### Task 8: Column insertion

**Files:**
- Modify: `src/qr_update.jl`, `test/qr_update.jl`

**Interfaces:**
- Consumes: `_grow!`, `_project!`, `_active`, `_spare`, `_clearspare!`, `shift_columns!(::UpdatableQR, i, j)`
- Produces: `insert_column!(F::UpdatableQR, j, x; rtol = sqrt(eps(real(T))))`

The `rtol` default is the Task 1 ruling. Written here for Option B.

- [ ] **Step 1: Write the failing test**

Append to `test/qr_update.jl`:

```julia
@testitem "QR column insertion, every index" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (8, 7), (20, 7), (30, 12))
        Afull = randn(T, m, n + 1)
        for j in 1:(n + 1)
            keep = setdiff(1:(n + 1), j)
            A = Afull[:, keep]
            x = Afull[:, j]
            F = UpdatableQR(A)
            insert_column!(F, j, x)
            @test size(F) == (m, n + 1)
            @test norm(F.Q * F.R - Afull) / norm(Afull) < 1.0e-12
            @test norm(F.Q' * F.Q - I) < 1.0e-12
            R = getfield(F, :factors)
            @test all(iszero, [R[i, j] for j in 1:F.n for i in (j + 1):F.n])
            @test all(iszero, view(R, (F.n + 1):size(R, 1), :))
        end
    end
end

@testitem "QR column insertion grows past its capacity" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity

    Random.seed!(20260908)
    m, n = 10, 4
    Afull = randn(m, n + 1)
    A = Afull[:, 1:n]
    F = UpdatableQR(A; capacity = (m, n))
    @test capacity(F) == (m, n)
    insert_column!(F, n + 1, Afull[:, n + 1])
    @test capacity(F) == (m, 2n)
    @test norm(F.Q * F.R - Afull) / norm(Afull) < 1.0e-12
    @test norm(F.Q' * F.Q - I) < 1.0e-12
end

@testitem "QR column insertion rejects a dependent column" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity

    Random.seed!(20260908)
    m, n = 9, 4
    A = randn(m, n)
    # A tight capacity puts the growth this verb would perform under the same assertion as the
    # rest of its state: a rejected insertion leaves the capacity where it was.
    F = UpdatableQR(A; capacity = (m, n))
    # An exact copy of an existing column leaves a residual of 1.95e-16, which is strictly
    # positive: it is the relative test against `norm(x)` that rejects it, not `rho > 0`.
    @test_throws "lies in the range of the existing columns" insert_column!(F, 2, A[:, 1])
    @test size(F) == (m, n)               # the throw left the factorization as it was
    @test capacity(F) == (m, n)
    @test norm(F.Q * F.R - A) / norm(A) < 1.0e-12
    qb = getfield(F, :qrep).buf
    @test all(iszero, view(qb, :, (F.n + 1):size(qb, 2)))

    # A rejected insertion leaves no residue for the next verb to build on. Without the clear,
    # this reconstruction measures 0.46 and this orthogonality 0.46, with nothing thrown.
    row = randn(n)
    insert_row!(F, 3, row)
    @test norm(F.Q' * F.Q - I) < 1.0e-12
    @test norm(F.Q * F.R - vcat(A[1:2, :], permutedims(row), A[3:m, :])) / norm(A) < 1.0e-12
end

@testitem "QR column insertion rtol widens the dependence test" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    m, n = 9, 4
    A = randn(m, n)
    x = A[:, 1] + 1.0e-6 .* randn(m)
    F = UpdatableQR(A)
    insert_column!(F, n + 1, x)
    @test size(F) == (m, n + 1)
    # The default admits the column, whose residual ratio is 5.9e-7, and leaves a measurably
    # collapsed diagonal entry.
    @test abs(F.R[n + 1, n + 1]) < 1.0e-4

    G = UpdatableQR(A)
    @test_throws "lies in the range of the existing columns" insert_column!(
        G, n + 1, x; rtol = 1.0e-4
    )
end

@testitem "QR column insertion rejects bad indices and a square factorization" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    F = UpdatableQR(randn(9, 4))
    @test_throws BoundsError insert_column!(F, 6, zeros(9))
    @test_throws BoundsError insert_column!(F, 0, zeros(9))
    @test_throws "x has length 8, factorization is 9x4" insert_column!(F, 1, zeros(8))
    G = UpdatableQR(randn(4, 4))
    @test_throws "the factorization requires m >= n" insert_column!(G, 1, zeros(4))
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — no `insert_column!` method for `UpdatableQR`.

- [ ] **Step 3: Write the implementation**

Append to `src/qr_update.jl`:

```julia
"""
    insert_column!(F::UpdatableQR, j, x; rtol = sqrt(eps(real(T)))) -> F

Insert `x` as column `j` of the factored matrix, in `O(mn + n|n + 1 - j|)` operations. `x` is
not modified.

The part of `x` orthogonal to the existing columns becomes the new column of `Q`, and its norm
`rho` becomes the new diagonal entry of `R`. `ArgumentError` is thrown when `rho` falls at or
below `rtol * norm(x)`: `rho` scales with `x`, so the threshold is relative, and the ratio it
tests lies in `[0, 1]`. A column that lies in the range of the existing ones leaves the
factorization exact but its last diagonal entry at the level of rounding noise, so a solve
through it divides by noise. `rtol = 0` admits every column whose residual is nonzero, which an
exactly dependent column is: measured, an exact copy of an existing column leaves `rho` at
1.95e-16 rather than at zero.

`DimensionMismatch` is thrown when the factorization is square, because the type admits only
`m >= n`.

Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating the
Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772-795.
"""
function insert_column!(
        F::UpdatableQR{T, S, <:DenseQ}, j::Integer, x::AbstractVector;
        rtol::Real = sqrt(eps(real(T)))
    ) where {T, S}
    m, n = F.m, F.n
    1 <= j <= n + 1 || throw(BoundsError(F, j))
    length(x) == m ||
        throw(DimensionMismatch("x has length $(length(x)), factorization is $(m)x$(n)"))
    m > n || throw(
        DimensionMismatch(
            "inserting a column would leave a $(m)x$(n + 1) factorization; " *
                "the factorization requires m >= n"
        )
    )
    q = getfield(F, :qrep)
    Qa = _active(q)
    r = _spare(q)
    w = view(F.work, 1:n)
    corr = view(F.corr, 1:n)
    ix = firstindex(x) - 1
    for i in 1:m
        r[i] = x[ix + i]
    end
    xnrm = norm(r)
    rho = _project!(w, r, Qa, corr)
    if !(rho > rtol * xnrm)
        # The projection built the candidate direction in the augmentation column. Clearing it
        # is what leaves the factorization as it was, and the next verb builds there.
        _clearspare!(q)
        throw(ArgumentError("column $j lies in the range of the existing columns: rho = $rho"))
    end
    r ./= rho
    # Growth comes after the guard, so a rejected insertion does not change the capacity.
    # `_grow!` carries the augmentation column into the new buffer, so the direction built
    # through `r` survives; the view itself does not, and nothing below reads it. `w` stays
    # valid because `_grow!` resizes `F.work` rather than rebinding it.
    _grow!(F, m, n + 1)
    R = getfield(F, :factors)
    for k in 1:n
        R[k, n + 1] = w[k]
    end
    R[n + 1, n + 1] = rho
    F.n = n + 1
    q.n = n + 1
    return j == n + 1 ? F : shift_columns!(F, n + 1, Int(j))
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS, five items.

- [ ] **Step 5: Commit**

```bash
git add src/qr_update.jl test/qr_update.jl
git commit -m "Add QR column insertion"
```

---

### Task 9: Row insertion

**Files:**
- Modify: `src/qr_update.jl`, `test/qr_update.jl`

**Interfaces:**
- Consumes: `_grow!`, `_insertrow!`, `_spare`, `_clearspare!`, `_raug`
- Produces: `insert_row!(F::UpdatableQR, i, x)`

- [ ] **Step 1: Write the failing test**

Append to `test/qr_update.jl`:

```julia
@testitem "QR row insertion, every index" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (8, 7), (6, 6), (9, 1), (30, 12))
        Afull = randn(T, m + 1, n)
        for i in 1:(m + 1)
            keep = setdiff(1:(m + 1), i)
            A = Afull[keep, :]
            x = Afull[i, :]
            F = UpdatableQR(A)
            insert_row!(F, i, x)
            @test size(F) == (m + 1, n)
            @test norm(F.Q * F.R - Afull) / norm(Afull) < 1.0e-12
            @test norm(F.Q' * F.Q - I) < 1.0e-12
            Q = getfield(F, :qrep)
            R = getfield(F, :factors)
            @test all(iszero, [R[i, j] for j in 1:F.n for i in (j + 1):F.n])
            @test all(iszero, view(Q.buf, :, (F.n + 1):size(Q.buf, 2)))
            @test all(iszero, view(R, (F.n + 1):size(R, 1), 1:n))
        end
    end
end

@testitem "QR row insertion grows the row capacity" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity

    Random.seed!(20260908)
    m, n = 8, 4
    Afull = randn(m + 1, n)
    A = Afull[1:m, :]
    F = UpdatableQR(A; capacity = (m, n))
    insert_row!(F, m + 1, Afull[m + 1, :])
    @test capacity(F) == (2m, n)
    @test norm(F.Q * F.R - Afull) / norm(Afull) < 1.0e-12
    @test norm(F.Q' * F.Q - I) < 1.0e-12
end

@testitem "QR row insertion rejects bad indices and lengths" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    F = UpdatableQR(randn(9, 4))
    @test_throws BoundsError insert_row!(F, 11, zeros(4))
    @test_throws BoundsError insert_row!(F, 0, zeros(4))
    @test_throws "x has length 3, factorization is 9x4" insert_row!(F, 1, zeros(3))
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — `insert_row!` has no methods.

- [ ] **Step 3: Write the implementation**

Append to `src/qr_update.jl`:

```julia
"""
    insert_row!(F::UpdatableQR, i, x) -> F

Insert `x` as row `i` of the factored matrix, in `O(mn + n^2)` operations. `x` is the new row,
not its adjoint, and is not modified.

A zero row opened in `Q` at position `i`, with the unit vector `e_i` as an extra column, extends
the factorization to the taller matrix; `n` rotations then return the appended row of `R` to
zero and the extra column of `Q` is dropped.

This is the verb that grows the row capacity, which re-strides every column of the stored
factor. A caller that inserts rows in a loop should pre-size with `capacity`.

Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5.
"""
function insert_row!(
        F::UpdatableQR{T, S, <:DenseQ}, i::Integer, x::AbstractVector
    ) where {T, S}
    m, n = F.m, F.n
    1 <= i <= m + 1 || throw(BoundsError(F, i))
    length(x) == n ||
        throw(DimensionMismatch("x has length $(length(x)), factorization is $(m)x$(n)"))
    # Grow before taking any view: growth rebinds both buffers.
    _grow!(F, m + 1, n)
    q = getfield(F, :qrep)
    _insertrow!(q, Int(i))
    R = getfield(F, :factors)
    RA = view(R, 1:(n + 1), 1:n)
    ix = firstindex(x) - 1
    for k in 1:n
        R[n + 1, k] = x[ix + k]
    end
    _spare(q)[i] = one(T)
    for k in 1:n
        c, s, rr = givensAlgorithm(R[k, k], R[n + 1, k])
        G = Givens(k, n + 1, oftype(R[k, k], c), oftype(R[k, k], s))
        lmul!(G, RA)
        rmul!(q, G')
        R[k, k] = rr
        R[n + 1, k] = zero(T)
    end
    _clearspare!(q)
    F.m = m + 1
    return F
end
```

The rotations pair the non-adjacent planes `(k, n+1)`; `LinearAlgebra.Givens` handles that, and `k < n + 1` always, so no inverted plane arises.

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS, three items.

- [ ] **Step 5: Commit**

```bash
git add src/qr_update.jl test/qr_update.jl
git commit -m "Add QR row insertion"
```

---

### Task 10: Row deletion and the leverage-one failure path

**Files:**
- Modify: `src/qr_update.jl`, `test/qr_update.jl`

**Interfaces:**
- Consumes: `_active`, `_augmented`, `_spare`, `_clearspare!`, `_deleterow!`, `_raug`, `_project_residual!`
- Produces: `delete_row!(F::UpdatableQR, i; rtol = sqrt(eps(real(T))))`

The ruling from Task 1 sets the default in the signature below. Written here for Option B.

- [ ] **Step 1: Write the failing test**

Append to `test/qr_update.jl`:

```julia
@testitem "QR row deletion, every index" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (8, 7), (9, 1), (30, 12))
        A = randn(T, m, n)
        @test norm(A' * A - I) > 1
        for i in 1:m
            F = UpdatableQR(A)
            delete_row!(F, i)
            keep = setdiff(1:m, i)
            @test size(F) == (m - 1, n)
            @test norm(F.Q * F.R - A[keep, :]) / norm(A) < 1.0e-12
            @test norm(F.Q' * F.Q - I) < 1.0e-12
            Q = getfield(F, :qrep)
            R = getfield(F, :factors)
            # The rotation sweep runs k descending so that R never leaves upper-triangular
            # form. An ascending sweep leaves Q*R exact and this subdiagonal at 0.75.
            @test all(iszero, [R[i, j] for j in 1:F.n for i in (j + 1):F.n])
            @test all(iszero, view(Q.buf, (F.m + 1):size(Q.buf, 1), :))
            @test all(iszero, view(Q.buf, :, (F.n + 1):size(Q.buf, 2)))
            @test all(iszero, view(R, (F.n + 1):(F.n + 1), 1:n))
        end
    end
end

@testitem "QR row deletion holds orthogonality across the leverage range" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    m, n = 10, 4
    thrown = 0
    for delta in (1.0e-1, 1.0e-3, 1.0e-5, 1.0e-7, 1.0e-9, 1.0e-12, 1.0e-15, 0.0)
        A = vcat(hcat(randn(m - 1, n - 1), delta .* randn(m - 1)), hcat(randn(1, n - 1), 1.0))
        F = UpdatableQR(A)
        try
            delete_row!(F, m)
        catch err
            err isa ArgumentError || rethrow()
            @test occursin("has leverage one", sprint(showerror, err))
            thrown += 1
            continue
        end
        @test norm(F.Q' * F.Q - I) < 1.0e-13
        @test norm(F.Q * F.R - A[1:(m - 1), :]) / norm(A) < 1.0e-12
    end
    # Both branches are reached: a sweep that only succeeds, or only throws, tests one of them.
    @test 0 < thrown < 8
end

@testitem "QR row deletion refuses a row of leverage one" begin
    using LinearAlgebra

    A = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    F = UpdatableQR(A)
    @test_throws "row 1 has leverage one" delete_row!(F, 1)
    @test size(F) == (3, 2)               # the throw left the factorization as it was
    @test norm(F.Q * F.R - A) / norm(A) < 1.0e-13
    qb = getfield(F, :qrep).buf
    @test all(iszero, view(qb, :, (F.n + 1):size(qb, 2)))
    delete_row!(F, 3)                     # a row with no leverage deletes cleanly
    @test size(F) == (2, 2)
    @test norm(F.Q * F.R - A[1:2, :]) / norm(A) < 1.0e-13
    @test norm(F.Q' * F.Q - I) < 1.0e-13
end

@testitem "QR row deletion rtol widens the leverage test" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    m, n = 10, 4
    A = vcat(hcat(randn(m - 1, n - 1), 1.0e-6 .* randn(m - 1)), hcat(randn(1, n - 1), 1.0))
    F = UpdatableQR(A)
    delete_row!(F, m)
    @test size(F) == (m - 1, n)
    # The default admits the deletion and leaves a measurably collapsed diagonal entry.
    @test minimum(abs, diag(F.R)) < 1.0e-4

    G = UpdatableQR(A)
    @test_throws "has leverage one" delete_row!(G, m; rtol = 1.0e-4)
end

@testitem "QR row deletion rejects a bad index and a square factorization" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    F = UpdatableQR(randn(9, 4))
    @test_throws BoundsError delete_row!(F, 10)
    @test_throws BoundsError delete_row!(F, 0)
    G = UpdatableQR(randn(4, 4))
    @test_throws "deleting row 2 would leave a 3x4 factorization" delete_row!(G, 2)
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — `delete_row!` has no methods.

- [ ] **Step 3: Write the implementation**

Append to `src/qr_update.jl`:

```julia
"""
    delete_row!(F::UpdatableQR, i; rtol = sqrt(eps(real(T)))) -> F

Remove row `i` of the factored matrix, in `O(mn)` operations.

The part of `e_i` orthogonal to the range of `Q` augments the factor to `m x (n+1)` columns, and
`n` rotations move row `i` of that augmented factor onto its last column, which is then dropped
along with the row. `gamma`, the norm of that orthogonal part, is the distance of `e_i` from the
range of `Q`: it lies in `[0, 1]`, and it agrees with the smallest singular value of the
remaining rows to several digits. `ArgumentError` is thrown, naming the row and `gamma`, when
`gamma <= rtol`; the deleted row then has leverage one and the numerical rank of what remains has
dropped, so the factorization that would be returned is a factorization of something else.

`gamma` is computed as the norm of the residual rather than from `sqrt(1 - norm(Q'e_i)^2)`. The
algebraic form subtracts nearly equal quantities: it reaches exactly zero for rows whose true
`gamma` is around `1e-15`, and goes negative around `1e-9`, so it refuses accurate deletions and
admits destructive ones.

`DimensionMismatch` is thrown when the factorization is square, because the type admits only
`m >= n`.

Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating the
Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772-795.
Reichel and Gragg, *Algorithm 686: FORTRAN subroutines for updating the QR decomposition*,
ACM Transactions on Mathematical Software 16 (1990), 369-377.
"""
function delete_row!(
        F::UpdatableQR{T, S, <:DenseQ}, i::Integer; rtol::Real = sqrt(eps(real(T)))
    ) where {T, S}
    m, n = F.m, F.n
    1 <= i <= m || throw(BoundsError(F, i))
    m > n || throw(
        DimensionMismatch(
            "deleting row $i would leave a $(m - 1)x$n factorization; " *
                "the factorization requires m >= n"
        )
    )
    q = getfield(F, :qrep)
    Qa = _active(q)
    t = view(F.work, 1:n)
    corr = view(F.corr, 1:n)
    z = _spare(q)
    for k in 1:n
        t[k] = conj(Qa[i, k])
    end
    mul!(z, Qa, t, -one(T), zero(T))
    z[i] += one(T)
    g = _project_residual!(t, z, Qa, corr)
    if !(g > rtol)
        # The projection built the augmentation direction in the spare column. Clearing it is
        # what leaves the factorization as it was, and the next verb builds there.
        _clearspare!(q)
        throw(
            ArgumentError(
                "row $i has leverage one: gamma = $g; deleting it drops the numerical rank"
            )
        )
    end
    z ./= g
    QA = _augmented(q)
    RA = _raug(F)
    for k in n:-1:1
        # Rotating row i of [Q z] on the right by G' sends its k-th entry to c*a + conj(s)*b,
        # with a = QA[i,k] and b = QA[i,n+1]; givensAlgorithm(-b, a) is the pair that makes it
        # vanish. Descending k keeps R upper triangular, so it is not retriangularized.
        a = QA[i, k]
        b = QA[i, n + 1]
        c, s, _ = givensAlgorithm(-b, a)
        G = Givens(k, n + 1, oftype(a, c), oftype(a, s))
        lmul!(G, RA)
        rmul!(q, G')
        QA[i, k] = zero(T)
    end
    _deleterow!(q, Int(i))
    for j in 1:n
        RA[n + 1, j] = zero(T)
    end
    F.m = m - 1
    return F
end
```

Descending `k` is load-bearing. At the start of step `k`, row `n+1` of `R` has support only in columns past `k`, so mixing it into row `k` fills strictly at or above `k`'s diagonal, and `R` never leaves upper-triangular form. Measured on the ascending mutant at `m,n = 10,5`: the subdiagonal of the stored block reaches 0.75, orthogonality stays at 1.4e-15, and `norm(Q * stored_block - target)` stays at 4.8e-16 — the fill is exactly compensated. Two assertions catch it, and both are needed: the triangularity check on the stored block, and `norm(F.Q * F.R - target)`, which reaches 0.11 because `F.R` wraps the block in `UpperTriangular` and discards the fill.

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS, five items.

- [ ] **Step 5: Commit**

```bash
git add src/qr_update.jl test/qr_update.jl
git commit -m "Add QR row deletion with its leverage test"
```

---

### Task 11: Invariants, allocation and type-stability gates

**Files:**
- Modify: `src/contracts.jl`, `test/strict.jl`

**Interfaces:**
- Consumes: all six verbs; `capacity(::UpdatableQR)`; `TypeContracts.behavior_passes`
- Produces: `@invariants UpdatableQR`; allocation and type-stability assertions

Task 1's Ruling 3 decides whether this task also adds `StrictMode` to `[deps]` and places `@assert_noalloc`/`@assert_typestable` inside the `DenseQ` rank-1 kernels. The `StrictModeTest` items below are written either way.

- [ ] **Step 1: Write the failing test**

Append to `test/strict.jl`:

```julia
@testitem "QR factorization invariants hold after every operation" begin
    using LinearAlgebra, Random, Test
    using UpdatableFactorizations: TypeContracts
    using .TypeContracts: behavior_passes

    Random.seed!(20260908)
    m, n = 12, 5
    A = randn(m, n)
    F = UpdatableQR(A)
    lowrankupdate!(F, randn(m), randn(n))
    insert_column!(F, 2, randn(m))
    delete_column!(F, 4)
    shift_columns!(F, 1, 4)
    insert_row!(F, 3, randn(F.n))
    delete_row!(F, 7)
    @test behavior_passes(UpdatableQR, [F])
end

@testitem "QR updating allocates nothing after construction" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    function measure(::Type{T}, m, n) where {T}
        mk() = UpdatableQR(randn(T, m, n))
        u = randn(T, m)
        v = randn(T, n)
        x = randn(T, m)
        row = randn(T, n)
        for F in (mk(), mk(), mk(), mk(), mk(), mk())   # compile every kernel before measuring
            lowrankupdate!(F, u, v)
            insert_column!(F, 2, x)
            delete_column!(F, 3)
            shift_columns!(F, 1, 4)
            insert_row!(F, 2, row)
            delete_row!(F, 5)
        end
        A, B, C, D, E, G = mk(), mk(), mk(), mk(), mk(), mk()
        return (
            @allocated(lowrankupdate!(A, u, v)),
            @allocated(insert_column!(B, 2, x)),
            @allocated(delete_column!(C, 3)),
            @allocated(shift_columns!(D, 1, 4)),
            @allocated(insert_row!(E, 2, row)),
            @allocated(delete_row!(G, 5)),
        )
    end
    @test measure(Float64, 80, 50) == (0, 0, 0, 0, 0, 0)
    @test measure(ComplexF64, 80, 50) == (0, 0, 0, 0, 0, 0)
end

@testitem "QR rank-1 kernels are type stable and allocate nothing" begin
    using LinearAlgebra, Random, StrictModeTest

    Random.seed!(20260908)
    for T in (Float64, ComplexF64)
        m, n = 40, 12
        F = UpdatableQR(randn(T, m, n))
        u = randn(T, m)
        v = randn(T, n)
        @test_typestable lowrankupdate!(F, u, v)
        @test_noalloc lowrankupdate!(F, u, v)
        G = UpdatableQR(randn(T, m, n))
        @test_typestable delete_row!(G, 3)
        H = UpdatableQR(randn(T, m, n))
        @test_noalloc delete_row!(H, 3)
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — `behavior_passes(UpdatableQR, [F])` errors because no behavior specs are registered for the type.

- [ ] **Step 3: Write the implementation**

Append to `src/contracts.jl`:

```julia
@invariants UpdatableQR begin
    # The body is parenthesized: an unparenthesized `;` splits the clause into two block
    # statements and the macro rejects the second as a spec that is not a pair.
    "size is within capacity" =>
        F -> ((mc, nc) = capacity(F); 0 <= F.n <= F.m <= mc && F.n <= nc)
    "the representation agrees with the factorization" =>
        F -> size(getfield(F, :qrep)) == (F.m, F.n)
    "workspaces cover the augmented block" =>
        F -> length(F.work) >= F.n + 1 && length(F.corr) >= F.n + 1
    "the active block of R is upper triangular" =>
        F -> all(iszero, [F.factors[i, j] for j in 1:F.n for i in (j + 1):F.n])
    "storage outside the active blocks is zero" =>
        F -> (Q = getfield(F, :qrep).buf; R = F.factors;
        all(iszero, view(Q, (F.m + 1):size(Q, 1), :)) &&
            all(iszero, view(Q, :, (F.n + 1):size(Q, 2))) &&
            all(iszero, view(R, (F.n + 1):size(R, 1), :)) &&
            all(iszero, view(R, :, (F.n + 1):size(R, 2))))
    # Comparing the Gram matrix against an explicit identity over the active columns is the
    # assertion itself, not a stand-in for iterating the factor's own indices.
    "Q has orthonormal columns" =>                                              # noidiom
        F -> (Q = F.Q; norm(Q' * Q - I) <= sqrt(eps(real(eltype(Q)))) * max(F.n, 1))
end
```

The `m >= n` clause is folded into the capacity chain rather than repeated: `0 <= F.n <= F.m <= mc` already asserts it.

There is deliberately no invariant on the sign or phase of `diag(R)`. `LinearAlgebra.qr` does not normalize it, nothing downstream reads it, and normalizing costs an `O(mn)` two-factor rescale on every verb. The invariant that replaces it is `norm(Q'Q - I)`.

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS, three items. If a `@allocated` entry is nonzero, find the allocation before relaxing the assertion: every buffer this milestone needs is a field, and a nonzero entry means a kernel took a view across a growth call or built a temporary.

- [ ] **Step 5: Commit**

```bash
git add src/contracts.jl test/strict.jl
git commit -m "Add UpdatableQR invariants and its allocation and stability gates"
```

---

### Task 12: Element types, derivatives and offset inputs

**Files:**
- Modify: `test/generic.jl`

**Interfaces:**
- Consumes: all six verbs
- Produces: no source change; coverage of `Float32`, `BigFloat`, `ForwardDiff.Dual`, `OffsetVector` and `view` inputs

- [ ] **Step 1: Calibrate the `Float32` tolerance before writing the item**

`100 * sqrt(eps(Float32))` is about `3.4e-2`, loose enough that a badly wrong kernel passes, so the committed item carries a measured literal rather than a placeholder to be tightened later.

In a session, run the six `Float32` operations of Step 2 and print the worst of their relative residuals. Then perturb one kernel — swap `conj` for identity in `lowrankupdate!`'s outer-product line — and print the worst again. The bound is a literal roughly an order of magnitude above the correct-case worst and below the perturbed minimum. Both measurements go into the comment, as `test/generic.jl` already does for the Cholesky loop:

```julia
        # BigFloat and Dual carry near-Float64 precision through generic arithmetic, so the
        # generic eps-scaled bound holds comfortably. Float32 rounding is coarser and its error
        # does not scale as cleanly with eps, so its bound is a value measured directly from
        # this test's own data (worst case over the six checks below is ~<measured>).
        tol = T === Float32 ? <measured literal> : 100 * sqrt(eps(real(T)))
```

Nothing is committed until both blanks hold numbers.

- [ ] **Step 2: Add the non-BLAS element-type item**

Append to `test/generic.jl`, with the `tol` line from Step 1 in place of the one below:

```julia
@testitem "QR verbs on non-BLAS element types" begin
    using LinearAlgebra, ForwardDiff, Random

    Random.seed!(3)
    DualT = ForwardDiff.Dual{Nothing, Float64, 1}

    m, n = 8, 4
    A64 = randn(m, n)
    u64 = randn(m)
    v64 = randn(n)
    x64 = randn(m)
    row64 = randn(n)

    for T in (Float32, BigFloat, DualT)
        A = T.(A64)
        u = T.(u64)
        v = T.(v64)
        x = T.(x64)
        row = T.(row64)
        # From Step 1. Left unparseable on purpose: the item is not committed until the
        # measured literal and the comment above it are in place.
        tol = T === Float32 ? <measured in Step 1> : 100 * sqrt(eps(real(T)))

        F = UpdatableQR(A)
        lowrankupdate!(F, u, v)
        @test norm(Matrix(F) - (A + u * v')) / norm(A) < tol

        G = UpdatableQR(A)
        insert_column!(G, 2, x)
        @test norm(Matrix(G) - hcat(A[:, 1], x, A[:, 2:n])) / norm(A) < tol

        H = UpdatableQR(A)
        delete_column!(H, 2)
        @test norm(Matrix(H) - A[:, setdiff(1:n, 2)]) / norm(A) < tol

        Sh = UpdatableQR(A)
        shift_columns!(Sh, 1, 3)
        @test norm(Matrix(Sh) - A[:, [2, 3, 1, 4]]) / norm(A) < tol

        Ir = UpdatableQR(A)
        insert_row!(Ir, 2, row)
        @test norm(Matrix(Ir) - vcat(A[1:1, :], permutedims(row), A[2:m, :])) / norm(A) < tol

        Dr = UpdatableQR(A)
        delete_row!(Dr, 2)
        @test norm(Matrix(Dr) - A[setdiff(1:m, 2), :]) / norm(A) < tol
        # Triangularity, asserted on the stored block: `F.R` wraps it in `UpperTriangular` and
        # reports a zero subdiagonal whatever the block holds.
        Rs = getfield(Dr, :factors)
        @test all(iszero, [Rs[i, j] for j in 1:Dr.n for i in (j + 1):Dr.n])

        # A Dual built by broadcasting a type constructor over Float64 values carries an
        # all-zero partial, so the checks above pass identically whether or not a kernel
        # actually propagates derivatives. Re-run each operation with one input seeded with a
        # nonzero partial and check that the result's partial is nonzero and finite.
        if T === DualT
            seed(y64, dir) = T.(y64, ForwardDiff.Partials.(tuple.(dir)))
            haspartial(M) = (
                ps = getindex.(ForwardDiff.partials.(M), 1);
                any(!iszero, ps) && all(isfinite, ps)
            )
            Aseed = seed(A64, randn(m, n))

            Fp = UpdatableQR(T.(A64))
            lowrankupdate!(Fp, seed(u64, randn(m)), v)
            @test haspartial(Matrix(Fp))

            Gp = UpdatableQR(T.(A64))
            insert_column!(Gp, 2, seed(x64, randn(m)))
            @test haspartial(Matrix(Gp))

            Hp = UpdatableQR(Aseed)
            delete_column!(Hp, 2)
            @test haspartial(Matrix(Hp))

            Sp = UpdatableQR(Aseed)
            shift_columns!(Sp, 1, 3)
            @test haspartial(Matrix(Sp))

            Ip = UpdatableQR(T.(A64))
            insert_row!(Ip, 2, seed(row64, randn(n)))
            @test haspartial(Matrix(Ip))

            Dp = UpdatableQR(Aseed)
            delete_row!(Dp, 2)
            @test haspartial(Matrix(Dp))
        end
    end
end
```

- [ ] **Step 3: Confirm each Dual case is not hollow**

For each of the six `haspartial` assertions, temporarily discard the seeded partial before the call (pass `T.(u64)` in place of `seed(u64, ...)`, or `T.(A64)` in place of `Aseed`) and confirm the assertion fails. A case that still passes is testing nothing; find an input whose derivative genuinely reaches the result and seed that one instead. Record which input each case seeds in a comment only if the choice is not obvious from the code.

- [ ] **Step 4: Add the offset-input item**

Append to `test/generic.jl`:

```julia
@testitem "QR verbs on offset and viewed inputs" begin
    using LinearAlgebra, OffsetArrays, Random

    Random.seed!(4)
    m, n = 8, 4
    A = randn(m, n)
    u = randn(m)
    v = randn(n)
    x = randn(m)
    row = randn(n)

    F = UpdatableQR(A)
    lowrankupdate!(F, u, v)
    G = UpdatableQR(A)
    # Different offsets on the two vectors: a shared offset lets a kernel that reuses one
    # origin variable for both of them pass.
    lowrankupdate!(G, OffsetVector(u, 0:(m - 1)), OffsetVector(v, -1:(n - 2)))
    @test Matrix(F) ≈ Matrix(G)
    H = UpdatableQR(A)
    lowrankupdate!(H, view(vcat(u, randn(3)), 1:m), view(vcat(v, randn(3)), 1:n))
    @test Matrix(F) ≈ Matrix(H)

    Fi = UpdatableQR(A)
    insert_column!(Fi, 2, x)
    Gi = UpdatableQR(A)
    insert_column!(Gi, 2, OffsetVector(x, -2:(m - 3)))
    @test Matrix(Fi) ≈ Matrix(Gi)
    Hi = UpdatableQR(A)
    insert_column!(Hi, 2, view(vcat(x, randn(3)), 1:m))
    @test Matrix(Fi) ≈ Matrix(Hi)

    Fr = UpdatableQR(A)
    insert_row!(Fr, 3, row)
    Gr = UpdatableQR(A)
    insert_row!(Gr, 3, OffsetVector(row, -1:(n - 2)))
    @test Matrix(Fr) ≈ Matrix(Gr)
    Hr = UpdatableQR(A)
    insert_row!(Hr, 3, view(vcat(row, randn(2)), 1:n))
    @test Matrix(Fr) ≈ Matrix(Hr)
end
```

`delete_column!`, `shift_columns!` and `delete_row!` take no vector argument, so they have no offset form. The solve path takes one and rejects it: `ldiv!` applies `Q'` to `b` in place and has no `m`-length scratch to copy an offset argument into, so it declares the limit with `Base.require_one_based_indexing`. Task 4's rejection item covers that.

- [ ] **Step 5: Run the tests to verify they pass**

Expected: PASS, two items. A failure in the offset item is an origin-arithmetic bug in `ix = firstindex(x) - 1`, not a tolerance problem.

- [ ] **Step 6: Commit**

```bash
git add test/generic.jl
git commit -m "Cover QR updating on non-BLAS element types and offset inputs"
```

---

### Task 13: Documentation and provenance

**Files:**
- Create: `docs/src/q_representations.md`
- Modify: `docs/make.jl`, `docs/src/updating.md`, `docs/src/provenance.md`

**Interfaces:**
- Consumes: the docstrings written in Tasks 2-10
- Produces: a built docs site including the Q-representations page and the extended provenance tables

- [ ] **Step 1: Check the citations against the articles**

The page and volume numbers carried in this plan's docstrings are transcribed, not verified. Open each article and confirm before the provenance page ships:

- Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating the Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772-795.
- Reichel and Gragg, *Algorithm 686: FORTRAN subroutines for updating the QR decomposition*, ACM Transactions on Mathematical Software 16 (1990), 369-377.
- Giraud, Langou and Rozložník, *The loss of orthogonality in the Gram-Schmidt orthogonalization process*, Computers and Mathematics with Applications 50 (2005), 1069-1075.
- Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5.

Correct any docstring that disagrees, in `src/`, before writing the table.

- [ ] **Step 2: Extend `docs/src/provenance.md`**

Add to the routines table:

```markdown
| QR rank-1 update | Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating the Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772-795 |
| QR column insert and row delete | Daniel, Gragg, Kaufman and Stewart, Mathematics of Computation 30 (1976), 772-795 |
| QR column delete and row insert | Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5 |
| QR column shift | Reichel and Gragg, *Algorithm 686: FORTRAN subroutines for updating the QR decomposition*, ACM Transactions on Mathematical Software 16 (1990), 369-377 |
| QR reorthogonalization | Giraud, Langou and Rozložník, *The loss of orthogonality in the Gram-Schmidt orthogonalization process*, Computers and Mathematics with Applications 50 (2005), 1069-1075 |
| Householder QR construction | `LinearAlgebra.qr`, which `qr_householder` calls; no algorithm is implemented here |
```

The "Reference implementations consulted" table and the `qrupdate-ng` statement stay as they are.

- [ ] **Step 3: Write `docs/src/q_representations.md`**

```markdown
# Q representations

`UpdatableQR` does not store its orthonormal factor directly. It holds an `AbstractQRep`, which
answers five operations: its shape, writing itself densely into a matrix, producing an explicitly
stored equivalent, and applying a Givens rotation on the left or on the right. A thin factor
cannot subtype `LinearAlgebra.AbstractQ`, whose interface is the implicit square factor —
`size(qr(randn(6, 3)).Q)` is `(6, 6)` — so the seam is local to this package.

## `DenseQ`

The one representation. It stores the thin `m x n` factor explicitly in the leading block of an
over-allocated buffer, with one further column that the updating verbs build their augmentation
in. `materialize` returns it unchanged.

`F.Q` is a view of that block. It changes when a verb changes the factorization, and writing
through it changes the factorization.

The docstrings for [`AbstractQRep`](@ref UpdatableFactorizations.AbstractQRep),
[`DenseQ`](@ref UpdatableFactorizations.DenseQ) and
[`materialize`](@ref UpdatableFactorizations.materialize) are on the API page.
```

The page carries no `@docs` block: `docs/src/api.md` is an `@autodocs` block over the whole
module and already documents every one of these names, so a second block here would be a
duplicate docstring and a build warning.

- [ ] **Step 4: Extend `docs/src/updating.md`**

Add a QR section to `updating.md` covering the six verbs, and stating four things plainly:

- `F.Q` and `F.R` are views of live storage, unlike `UpdatableLU`'s reassembled factors.
- `delete_row!` and `insert_column!` throw when the operation would leave `m < n`.
- `delete_row!` throws when the deleted row has leverage one, and `insert_column!` throws when the inserted column lies in the range of the existing ones; both messages name the measured quantity.
- Every update reorthogonalizes unconditionally. That costs a second `O(mn)` pair of matrix-vector products in `lowrankupdate!`, `insert_column!` and `delete_row!`, and it is what keeps `norm(Q'Q - I)` at the level of rounding across the whole leverage range. The package is slower than recomputing with LAPACK for small factorizations, and this pass widens that gap.

Note there that `capacity` for `UpdatableQR` returns the pair `(mcap, ncap)`, that row growth re-strides the stored factor while column growth does not, and that `qr_householder` is the recommended way to build a factorization to update: its orthogonality is at machine precision whatever the condition number.

`docs/src/api.md` is not edited. After the build in Step 6, confirm `UpdatableQR`, `qr_householder`, `insert_row!`, `delete_row!`, `AbstractQRep`, `DenseQ`, `materialize` and `capacity` all appear on the API page; `@autodocs` includes a `public` but unexported name, so a missing one points at the `public` declaration in `src/UpdatableFactorizations.jl`.

- [ ] **Step 5: Register the page**

`docs/make.jl`, in `pages`, after `"Updating and downdating"`:

```julia
        "Q representations" => "q_representations.md",
```

- [ ] **Step 6: Build the docs**

Run `julia --project=docs docs/make.jl` in a session with the package developed into the docs environment.
Expected: builds with no missing-docstring and no broken-cross-reference warnings.

- [ ] **Step 7: Commit**

```bash
git add docs/
git commit -m "Document QR updating and its Q representation"
```

---

### Task 14: Full-suite gate

**Files:**
- Modify: none expected

- [ ] **Step 1: Run the whole suite**

`julia_run_testitems` over the package with `max_workers = 4`, no filter.
Expected: every item passes, including the M0/M2a items.

- [ ] **Step 2: Run `Pkg.test()` once, cold**

This is the pre-commit gate, and the only place a cold Julia process is warranted. The warm runner masks a missing test dependency, which is why this step exists.
Expected: passes.

- [ ] **Step 3: Format**

Run `runic` over `src/` and `test/`. Commit any reformatting separately.

- [ ] **Step 4: Check the diff's comments and docstrings**

Re-read every comment and docstring added by this milestone. Any that references this plan, a milestone, a task number, a judge, a candidate design, or how the code came to be is a defect; rewrite it to state what is true of the code now. Check in particular that no docstring or doc page claims a speed advantage over LAPACK.

- [ ] **Step 5: Record the divergences in the ledger**

Add each of these, with the measurement or reasoning that settled it:

1. `work::Matrix{T}` in the spec's `UpdatableQR` sketch became two `Vector{T}` fields plus the spare column of the `Q` buffer; no verb needs `m`-length scratch once the augmentation column is storage. The spec's "the augmented column of section 4.4" is a dangling cross-reference — the spec has no section 4.4.
2. The contract's `Matrix` clause is spelled `copyto!(::AbstractMatrix, ::Self)`. TypeContracts 0.14 cannot register a constructor in a contract; `Base.Matrix` is still defined once, on the abstract type. This is an expressibility limit, not a design change.
3. No `det`, `logdet` or `logabsdet`: `LinearAlgebra` defines none for a QR factorization, and a thin `Q` does not determine the sign.
4. `capacity` returns a tuple for `UpdatableQR` and becomes `public`. `capacity(::UpdatableCholesky)` is unchanged.
5. `insert_column!` gains an `rtol` keyword on the QR method, which the Cholesky method does not have.
6. `AbstractQRep` deliberately does not subtype `LinearAlgebra.AbstractQ`.
7. `StrictMode` in `[deps]`, as ruled in Task 1. Record the ruling and, under Option B, the spec amendment to R10 that goes with it.
8. `delete_row!`'s and `insert_column!`'s default `rtol`, as ruled in Task 1.
9. The updating verbs accept offset vectors and the solve path does not. The verbs copy their argument into the factorization's storage through `ix = firstindex(x) - 1`, and the offset item in `test/generic.jl` is the defense. `ldiv!` applies `Q'` to `b` in place, has no `m`-length scratch to copy into, and so declares the limit with `Base.require_one_based_indexing` on both methods; Task 4's rejection item is the defense there.
10. The `UpdatableQR` constructor rejects only an exactly zero diagonal entry, not a numerically dependent column. `insert_column!`'s `rtol` is the relative test, and the docstring says so.
11. `UpdatableQR` gains a two-matrix constructor, `UpdatableQR(Q, R; capacity)`, which the spec does not list. It is the one place the storage layout is built from an already-computed thin factorization, and the construction layer's `qr_bcgs` returns through it as well.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Format and finish QR updating"
```

---

## Self-review

**Spec coverage for this milestone.** R5 (Task 3's `qr_householder`, which delegates to `LinearAlgebra.qr` and wraps the result; it lands here because `UpdatableQR`, the type it returns, is built here), R7's QR half (Tasks 5-10, six verbs), R9 (Task 2's `@contract` on `AbstractQRep`, Task 11's `@invariants UpdatableQR`, whose capacity clause carries the `m >= n` assertion), R11 (Task 12), R15 (Tasks 2-10 docstrings, Task 13 provenance). R2-R4, R6 and R14 belong to M1, R8 to M3, R12's benchmark page and R13 to M4.

R10 has two halves — "StrictMode assertions on `DenseQ` rank-1 kernels" and "StrictModeTest gating CI" — and Task 11 as written ships only the second. Task 1's Ruling 3 decides whether the first is built or the requirement is amended. R10 is not claimed as covered until that ruling exists.

**Known costs, stated rather than smoothed over.**

- The default `capacity = (2m, 2n)` makes the `Q` buffer four times the thin factor, plus a `(2n+1)²` `R` buffer. A caller who inserts rarely should pass `capacity = (m, n)`, which pre-sizes exactly.
- `insert_row!` past `mcap` re-strides every stored column. There is no way to make that cheap in a column-major layout; doubling makes it amortized, and the asymmetry with column growth is documented because the default `2m` hides it in benchmarks.
- The reorthogonalization pass is unconditional, costing a second `O(mn)` pair of matrix-vector products in three verbs. The conditional criterion leaves `eps/gamma` behavior on the table, and orthogonality is a stated invariant, so the pass stays. It widens the measured gap against recomputing with LAPACK, and no document may claim otherwise.
- `AbstractQRep` has one implementation until M3. The verbs dispatch on `<:DenseQ` and reach storage through `_active`/`_augmented`/`_spare`, which are not contract methods. If M3 slips, this is one abstract type, one struct and one `@contract` block to delete.
- Non-BLAS element types get correctness coverage, not an allocation guarantee. The generic `mul!` fallback for `BigFloat` and `Dual` is not measured, and the `@allocated == 0` gate runs on `Float64` and `ComplexF64` only.

**The one thing that must not be relaxed.** `delete_row!`'s `gamma` is the norm of the computed residual, and its reorthogonalization pass is not optional. Measured on a leverage-controlled fixture, the algebraic form `sqrt(1 - norm(Q'e_i)^2)` refuses deletions accurate to 2e-16 and admits ones that drive `norm(Q'Q - I)` to 0.97, while the reconstruction residual stays at 1.7e-16 in every cell. A reconstruction test is blind to this, exactly as it was blind to the Cholesky diagonal-sign defect. The orthogonality invariant and the leverage sweep are what catch it, and neither may be narrowed to make a failure go away.
