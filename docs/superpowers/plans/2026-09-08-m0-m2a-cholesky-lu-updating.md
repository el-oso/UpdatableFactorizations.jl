# M0 + M2a: package skeleton and Cholesky/LU updating — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a registered-quality Julia package whose `UpdatableCholesky` and `UpdatableLU` types support every Cholesky and LU modification in `qrupdate`'s catalogue, including the LU rank-1 update that exists nowhere else in pure Julia.

**Architecture:** Two mutable `LinearAlgebra.Factorization` subtypes hold their factors in over-allocated buffers with an active leading block, so insertions do not reallocate. Cholesky work is expressed once against a lower-triangular view — for `uplo = 'U'` that view is the storage's adjoint — so a single kernel serves both orientations. LU is stored in `LDU` form because Bennett's rank-1 update is natural there, with `F.L` and `F.U` reassembled on property access to match the standard library.

**Tech Stack:** Julia 1.12, `LinearAlgebra`, `TypeContracts` 0.14, `StrictMode` 0.4.1, `StrictModeTest` 0.4.1 (test only), `TestItems`/`TestItemRunner`, DocumenterVitepress.

**Spec:** `docs/superpowers/specs/2026-09-08-updatablefactorizations-design.md`

**Verified reference implementation:** `docs/superpowers/specs/proto.jl` and `verify.jl`. Every kernel in this plan was run against a from-scratch recomputed factorization for both element types, both `uplo`, and every index, before the plan was written. The code blocks below are that verified code, renamed. Do not "improve" a kernel without re-running `verify.jl`.

## Global Constraints

- License MIT. No GPL code in the dependency graph outside `bench/`. `QRupdatesFast` is GPL-linked and may appear only in `bench/Project.toml`.
- `julia = "1.12"`. `StrictMode = "0.4.1"`, `TypeContracts = "0.14"`.
- Never read `qrupdate-ng` source. It is a benchmark target and routine-list reference only.
- Every algorithm's docstring cites its source article (spec section 6).
- No `@inbounds` anywhere in this plan.
- Comments state what is true of the code now. No references to this plan, to milestones, or to how the code came to be.
- Run tests with the JuliaMCP `julia_run_testitems` tool, always passing `max_workers`. Do not shell out to `Pkg.test()` except as the final pre-commit gate.
- American spellings.

## File Structure

| file | responsibility |
| --- | --- |
| `Project.toml` | package metadata, deps, compat |
| `src/UpdatableFactorizations.jl` | module, includes, exports |
| `src/cholesky_type.jl` | `UpdatableCholesky`, `_lower`, constructors, `Factorization` interface |
| `src/cholesky_update.jl` | `lowrankupdate!`, `lowrankdowndate!` for Cholesky |
| `src/cholesky_resize.jl` | `insert_column!`, `delete_column!`, `shift_columns!`, `_lq!`, `_permute!` |
| `src/lu_type.jl` | `UpdatableLU`, constructors, `getproperty`, `Factorization` interface |
| `src/lu_update.jl` | `lowrankupdate!` for LU (Bennett) |
| `src/contracts.jl` | `@invariants` on both types |
| `test/*.jl` | one file per source file, `@testitem` per routine |
| `docs/` | DocumenterVitepress site, including the provenance page |

---

### Task 1: Package skeleton

**Files:**
- Create: `Project.toml`, `LICENSE`, `README.md`, `src/UpdatableFactorizations.jl`, `test/Project.toml`, `test/runtests.jl`, `.github/workflows/CI.yml`

**Interfaces:**
- Consumes: nothing
- Produces: module `UpdatableFactorizations`; a test suite runnable via TestItemRunner

- [ ] **Step 1: Create `Project.toml`**

Generate the UUID with `using UUIDs; uuid4()` — never hand-write one.

```toml
name = "UpdatableFactorizations"
uuid = "<generated with UUIDs.uuid4()>"
version = "0.1.0"
authors = ["el_oso"]

[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
StrictMode = "0e98b4f4-d0e6-42ac-af0a-707c24852f32"
TypeContracts = "c7a3f1e2-84b5-4d69-9e0a-1f2b3c4d5e6f"

[compat]
LinearAlgebra = "1.12"
StrictMode = "0.4.1"
TypeContracts = "0.14"
julia = "1.12"
```

- [ ] **Step 2: Create `src/UpdatableFactorizations.jl`**

```julia
module UpdatableFactorizations

using LinearAlgebra
using LinearAlgebra: givensAlgorithm, PosDefException, ZeroPivotException
import LinearAlgebra: lowrankupdate!, lowrankdowndate!, ldiv!, logdet, det

export UpdatableCholesky, UpdatableLU
export insert_column!, delete_column!, shift_columns!

include("cholesky_type.jl")
include("cholesky_update.jl")
include("cholesky_resize.jl")
include("lu_type.jl")
include("lu_update.jl")
include("contracts.jl")

end
```

Create the five included files empty for now so the module loads.

- [ ] **Step 3: Create `test/Project.toml` and `test/runtests.jl`**

```toml
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
StrictModeTest = "<look up with Pkg, do not guess>"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
TestItemRunner = "f8b46487-2199-4994-9208-9a1283c18c0a"
```

Add every dependency with `Pkg.add` from inside `test/`, never by hand-writing a UUID. Replace the placeholder above with what `Pkg` writes.

```julia
using TestItemRunner
@run_package_tests
```

- [ ] **Step 4: Verify the package loads and the empty suite runs**

Use `julia_create_session` with `project` set to the package root, then evaluate `using UpdatableFactorizations`.
Expected: loads with no error. `julia_run_testitems` with `max_workers = 4` reports 0 test items.

- [ ] **Step 5: Add `LICENSE` (MIT, copyright el_oso, 2026) and a `README.md`**

The README states what the package does, that it extends `LinearAlgebra.lowrankupdate!`/`lowrankdowndate!` rather than replacing them, and links the provenance page. It makes no speed claim.

- [ ] **Step 6: Add `.github/workflows/CI.yml`**

Julia 1.12 on ubuntu-latest, running the test suite and uploading coverage to Coveralls (tokenless). Follow the workflow already used by `StrictMode.jl` in this ecosystem.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Add package skeleton"
```

---

### Task 2: `UpdatableCholesky` type

**Files:**
- Create: `src/cholesky_type.jl`, `test/cholesky_type.jl`

**Interfaces:**
- Consumes: nothing
- Produces: `UpdatableCholesky{T,S}` with fields `factors::S`, `n::Int`, `uplo::Char`, `work::Vector{T}`; `_lower(F)`; `UpdatableCholesky(::Cholesky)`; `UpdatableCholesky(A::AbstractMatrix; uplo, capacity)`; `size`, `Matrix`, `ldiv!`, `\`, `logdet`, `det`

- [ ] **Step 1: Write the failing test**

`test/cholesky_type.jl`:

```julia
@testitem "UpdatableCholesky reconstructs its matrix" begin
    using LinearAlgebra
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        B = randn(T, 6, 6)
        A = Matrix(Hermitian(B * B' + 6I))
        F = UpdatableCholesky(cholesky(Hermitian(A, uplo)))
        L = Matrix(F)
        @test norm(L * L' - A) / norm(A) < 1e-13
        @test size(F) == (6, 6)
    end
end

@testitem "UpdatableCholesky ignores the unstored triangle" begin
    using LinearAlgebra
    B = randn(6, 6)
    A = Matrix(Symmetric(B * B' + 6I))
    C = cholesky(Symmetric(A, :L))
    # cholesky leaves the original matrix in the strict upper triangle; it must not leak in.
    @test any(!iszero, [C.factors[i, j] for i in 1:6 for j in (i + 1):6])
    F = UpdatableCholesky(C)
    @test all(iszero, [F.factors[i, j] for i in 1:6 for j in (i + 1):6])
end

@testitem "UpdatableCholesky solves" begin
    using LinearAlgebra
    B = randn(6, 6)
    A = Matrix(Symmetric(B * B' + 6I))
    b = randn(6)
    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    @test norm(A * (F \ b) - b) / norm(b) < 1e-12
end
```

- [ ] **Step 2: Run the tests to verify they fail**

`julia_run_testitems` filtered to `cholesky_type`, `max_workers = 4`.
Expected: FAIL — `UpdatableCholesky` not defined.

- [ ] **Step 3: Write the implementation**

`src/cholesky_type.jl`:

```julia
"""
    UpdatableCholesky(C::Cholesky; capacity = 2size(C, 1))
    UpdatableCholesky(A::AbstractMatrix; uplo = :L, capacity = 2size(A, 1))

Cholesky factorization that supports rank-1 update and downdate and symmetric insertion,
deletion and shifting of indices. `capacity` is the largest size the factorization can reach
before its storage is reallocated.
"""
mutable struct UpdatableCholesky{T, S <: AbstractMatrix{T}} <: Factorization{T}
    factors::S
    n::Int
    uplo::Char
    work::Vector{T}
end

function UpdatableCholesky(C::Cholesky{T}; capacity::Int = 2size(C, 1)) where {T}
    n = size(C, 1)
    capacity >= n || throw(ArgumentError("capacity $capacity is below the size $n"))
    f = zeros(T, capacity, capacity)
    # Cholesky.factors only guarantees the stored triangle; LAPACK leaves the factored matrix in
    # the other one. Copying the stored triangle alone keeps the unstored half a true zero,
    # which the resizing kernels rely on.
    if C.uplo == 'L'
        for j in 1:n, i in j:n
            f[i, j] = C.factors[i, j]
        end
    else
        for j in 1:n, i in 1:j
            f[i, j] = C.factors[i, j]
        end
    end
    return UpdatableCholesky{T, Matrix{T}}(f, n, C.uplo, zeros(T, capacity))
end

UpdatableCholesky(A::AbstractMatrix; uplo::Symbol = :L, capacity::Int = 2size(A, 1)) =
    UpdatableCholesky(cholesky(Hermitian(A, uplo)); capacity)

# The active factor presented as lower triangular. When the storage holds U with A = U'U, its
# adjoint is the lower factor L = U', and assignments through it conjugate as they must.
_lower(F::UpdatableCholesky) =
    F.uplo == 'L' ? view(F.factors, 1:F.n, 1:F.n) : adjoint(view(F.factors, 1:F.n, 1:F.n))

Base.size(F::UpdatableCholesky) = (F.n, F.n)
Base.size(F::UpdatableCholesky, i::Integer) = i <= 2 ? F.n : 1
Base.Matrix(F::UpdatableCholesky) = LowerTriangular(Matrix(_lower(F)))

capacity(F::UpdatableCholesky) = size(F.factors, 1)

function LinearAlgebra.ldiv!(F::UpdatableCholesky, b::AbstractVecOrMat)
    L = LowerTriangular(_lower(F))
    ldiv!(L, b)
    ldiv!(L', b)
    return b
end

Base.:\(F::UpdatableCholesky, b::AbstractVecOrMat) = ldiv!(F, copy(b))

LinearAlgebra.logdet(F::UpdatableCholesky) = 2 * sum(i -> log(real(_lower(F)[i, i])), 1:F.n)
LinearAlgebra.det(F::UpdatableCholesky) = exp(logdet(F))
```

- [ ] **Step 4: Run the tests to verify they pass**

`julia_run_testitems` filtered to `cholesky_type`, `max_workers = 4`.
Expected: PASS, all three items.

- [ ] **Step 5: Commit**

```bash
git add src/cholesky_type.jl test/cholesky_type.jl
git commit -m "Add UpdatableCholesky and its solve"
```

---

### Task 3: Cholesky rank-1 update

**Files:**
- Create: `src/cholesky_update.jl`, `test/cholesky_update.jl`

**Interfaces:**
- Consumes: `UpdatableCholesky`, `_lower` (Task 2)
- Produces: `LinearAlgebra.lowrankupdate!(F::UpdatableCholesky, v)`; internal `_ch1up!(L, w)` which the resizing kernels reuse

- [ ] **Step 1: Write the failing test**

`test/cholesky_update.jl`:

```julia
@testitem "Cholesky rank-1 update" begin
    using LinearAlgebra
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        B = randn(T, 7, 7)
        A = Matrix(Hermitian(B * B' + 7I))
        v = randn(T, 7)
        F = UpdatableCholesky(cholesky(Hermitian(A, uplo)))
        lowrankupdate!(F, v)
        L = Matrix(F)
        @test norm(L * L' - (A + v * v')) / norm(A) < 1e-13
    end
end

@testitem "Cholesky rank-1 update does not consume its vector" begin
    using LinearAlgebra
    B = randn(6, 6)
    A = Matrix(Symmetric(B * B' + 6I))
    v = randn(6)
    vcopy = copy(v)
    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    lowrankupdate!(F, v)
    @test v == vcopy
end

@testitem "Cholesky rank-1 update rejects a mismatched vector" begin
    using LinearAlgebra
    B = randn(6, 6)
    F = UpdatableCholesky(cholesky(Symmetric(Matrix(Symmetric(B * B' + 6I)), :L)))
    @test_throws "has length 5, factorization is 6" lowrankupdate!(F, randn(5))
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — no method `lowrankupdate!` for `UpdatableCholesky`.

- [ ] **Step 3: Write the implementation**

`src/cholesky_update.jl`:

```julia
"""
    lowrankupdate!(F::UpdatableCholesky, v) -> F

Replace the factorization of `A` with that of `A + v*v'` in `O(n^2)` operations. `v` is not
modified. Extends `LinearAlgebra.lowrankupdate!`.

Gill, Golub, Murray and Saunders, *Methods for modifying matrix factorizations*,
Mathematics of Computation 28 (1974), 505-535.
"""
function LinearAlgebra.lowrankupdate!(F::UpdatableCholesky, v::AbstractVector)
    length(v) == F.n ||
        throw(DimensionMismatch("v has length $(length(v)), factorization is $(F.n)"))
    w = view(F.work, 1:F.n)
    copyto!(w, v)
    _ch1up!(_lower(F), w)
    return F
end

# Chase a rank-1 term into a lower-triangular factor with Givens rotations. `w` is consumed.
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS, all three items.

- [ ] **Step 5: Commit**

```bash
git add src/cholesky_update.jl test/cholesky_update.jl
git commit -m "Add Cholesky rank-1 update"
```

---

### Task 4: Cholesky rank-1 downdate

**Files:**
- Modify: `src/cholesky_update.jl`, `test/cholesky_update.jl`

**Interfaces:**
- Consumes: `UpdatableCholesky`, `_lower`
- Produces: `LinearAlgebra.lowrankdowndate!(F::UpdatableCholesky, v)`

- [ ] **Step 1: Write the failing test**

Append to `test/cholesky_update.jl`:

```julia
@testitem "Cholesky rank-1 downdate" begin
    using LinearAlgebra
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        B = randn(T, 7, 7)
        A = Matrix(Hermitian(B * B' + 7I))
        v = randn(T, 7) ./ 8
        F = UpdatableCholesky(cholesky(Hermitian(A, uplo)))
        lowrankdowndate!(F, v)
        L = Matrix(F)
        @test norm(L * L' - (A - v * v')) / norm(A) < 1e-13
    end
end

@testitem "Cholesky downdate past positive definiteness throws" begin
    using LinearAlgebra
    B = randn(6, 6)
    A = Matrix(Symmetric(B * B' + 6I))
    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    # A - v*v' with v far larger than any direction of A cannot be positive definite.
    @test_throws PosDefException lowrankdowndate!(F, randn(6) .* 1000)
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — no method `lowrankdowndate!` for `UpdatableCholesky`.

- [ ] **Step 3: Write the implementation**

Append to `src/cholesky_update.jl`:

```julia
"""
    lowrankdowndate!(F::UpdatableCholesky, v) -> F

Replace the factorization of `A` with that of `A - v*v'` in `O(n^2)` operations. `v` is not
modified. Throws `PosDefException` when the result would not be positive definite. Extends
`LinearAlgebra.lowrankdowndate!`.

Uses the mixed formulation of Bojanczyk, Brent, Van Dooren and de Hoog, *A note on downdating
the Cholesky factorization*, SIAM Journal on Scientific and Statistical Computing 8 (1987),
210-221, which is more stable near breakdown than forming the hyperbolic rotation directly.
"""
function LinearAlgebra.lowrankdowndate!(F::UpdatableCholesky, v::AbstractVector)
    length(v) == F.n ||
        throw(DimensionMismatch("v has length $(length(v)), factorization is $(F.n)"))
    w = view(F.work, 1:F.n)
    copyto!(w, v)
    _ch1dn!(_lower(F), w)
    return F
end

# `w` is consumed, first as the right-hand side of L p = w and then as p itself.
function _ch1dn!(L, w)
    n = size(L, 1)
    T = eltype(L)
    R = real(T)
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cholesky_update.jl test/cholesky_update.jl
git commit -m "Add Cholesky rank-1 downdate"
```

---

### Task 5: Cholesky index deletion

**Files:**
- Create: `src/cholesky_resize.jl`, `test/cholesky_resize.jl`

**Interfaces:**
- Consumes: `UpdatableCholesky`, `_lower`, `_ch1up!`
- Produces: `delete_column!(F::UpdatableCholesky, j)`

- [ ] **Step 1: Write the failing test**

`test/cholesky_resize.jl`:

```julia
@testitem "Cholesky symmetric deletion, every index" begin
    using LinearAlgebra
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        n = 7
        B = randn(T, n, n)
        A = Matrix(Hermitian(B * B' + n * I))
        for j in 1:n
            F = UpdatableCholesky(cholesky(Hermitian(A, uplo)))
            delete_column!(F, j)
            keep = [k for k in 1:n if k != j]
            L = Matrix(F)
            @test size(F) == (n - 1, n - 1)
            @test norm(L * L' - A[keep, keep]) / norm(A) < 1e-13
        end
    end
end

@testitem "Cholesky deletion rejects an out-of-range index" begin
    using LinearAlgebra
    B = randn(5, 5)
    F = UpdatableCholesky(cholesky(Symmetric(Matrix(Symmetric(B * B' + 5I)), :L)))
    @test_throws BoundsError delete_column!(F, 6)
    @test_throws BoundsError delete_column!(F, 0)
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — `delete_column!` not defined.

- [ ] **Step 3: Write the implementation**

`src/cholesky_resize.jl`:

```julia
"""
    delete_column!(F::UpdatableCholesky, j) -> F

Remove index `j`, deleting both row `j` and column `j` of the factored matrix, in `O(n^2)`
operations.

Writing the factor in blocks about `j`, the leading block and the block below and to the left of
`j` are already the factors of the reduced matrix. Only the trailing block changes, and it
changes by a rank-1 update with the deleted column's tail.

Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5.
"""
function delete_column!(F::UpdatableCholesky, j::Integer)
    n = F.n
    1 <= j <= n || throw(BoundsError(F, j))
    L = _lower(F)
    tail = [L[i, j] for i in (j + 1):n]
    for c in 1:(n - 1)
        for r in 1:(n - 1)
            L[r, c] = L[r + (r >= j), c + (c >= j)]
        end
    end
    for k in 1:n                       # clear the vacated last row and column
        L[n, k] = zero(eltype(L))
        L[k, n] = zero(eltype(L))
    end
    F.n = n - 1
    isempty(tail) || _ch1up!(view(_lower(F), j:(n - 1), j:(n - 1)), tail)
    return F
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cholesky_resize.jl test/cholesky_resize.jl
git commit -m "Add Cholesky symmetric index deletion"
```

---

### Task 6: Cholesky append and capacity growth

**Files:**
- Modify: `src/cholesky_resize.jl`, `test/cholesky_resize.jl`

**Interfaces:**
- Consumes: `UpdatableCholesky`, `_lower`, `capacity`
- Produces: internal `_append!(F, x)`; `_grow!(F, needed)`

- [ ] **Step 1: Write the failing test**

Append to `test/cholesky_resize.jl`:

```julia
@testitem "Cholesky append at the end" begin
    using LinearAlgebra
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        n = 6
        B = randn(T, n + 1, n + 1)
        A = Matrix(Hermitian(B * B' + (n + 1) * I))
        F = UpdatableCholesky(cholesky(Hermitian(A[1:n, 1:n], uplo)))
        UpdatableFactorizations._append!(F, A[:, n + 1])
        L = Matrix(F)
        @test size(F) == (n + 1, n + 1)
        @test norm(L * L' - A) / norm(A) < 1e-13
    end
end

@testitem "Cholesky append grows past capacity" begin
    using LinearAlgebra
    n = 4
    B = randn(n + 1, n + 1)
    A = Matrix(Symmetric(B * B' + (n + 1) * I))
    F = UpdatableCholesky(cholesky(Symmetric(A[1:n, 1:n], :L)); capacity = n)
    UpdatableFactorizations._append!(F, A[:, n + 1])
    L = Matrix(F)
    @test norm(L * L' - A) / norm(A) < 1e-13
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — `_append!` not defined.

- [ ] **Step 3: Write the implementation**

Append to `src/cholesky_resize.jl`:

```julia
function _grow!(F::UpdatableCholesky{T}, needed::Int) where {T}
    needed <= capacity(F) && return F
    newcap = max(needed, 2capacity(F))
    f = zeros(T, newcap, newcap)
    copyto!(view(f, 1:F.n, 1:F.n), view(F.factors, 1:F.n, 1:F.n))
    F.factors = f
    resize!(F.work, newcap)
    return F
end

# Append a new last index. `x` has length n+1, with x[n+1] the new diagonal entry.
function _append!(F::UpdatableCholesky{T}, x::AbstractVector) where {T}
    n = F.n
    length(x) == n + 1 ||
        throw(DimensionMismatch("x has length $(length(x)), expected $(n + 1)"))
    L = _lower(F)
    l = Vector{T}(undef, n)
    for i in 1:n
        acc = x[i]
        for k in 1:(i - 1)
            acc -= L[i, k] * l[k]
        end
        l[i] = acc / L[i, i]
    end
    d = real(x[n + 1]) - sum(abs2, l)
    d > 0 || throw(PosDefException(n + 1))
    _grow!(F, n + 1)
    F.n = n + 1
    M = _lower(F)
    for k in 1:n
        M[n + 1, k] = conj(l[k])
        M[k, n + 1] = zero(T)
    end
    M[n + 1, n + 1] = sqrt(d)
    return F
end
```

Note `F.factors` is reassigned by `_grow!`, so `_lower(F)` must be recomputed after it; the code
above does that by binding `M` only after growing.

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cholesky_resize.jl test/cholesky_resize.jl
git commit -m "Add Cholesky append with capacity growth"
```

---

### Task 7: Cholesky shift and arbitrary-index insertion

**Files:**
- Modify: `src/cholesky_resize.jl`, `test/cholesky_resize.jl`

**Interfaces:**
- Consumes: `_append!`, `_lower`
- Produces: `_lq!(B)`; `_permute!(F, perm)`; `_cyclicperm(n, i, j)`; `shift_columns!(F, i, j)`; `insert_column!(F, j, x)`

- [ ] **Step 1: Write the failing test**

Append to `test/cholesky_resize.jl`:

```julia
@testitem "Cholesky index shift, both directions" begin
    using LinearAlgebra
    n = 7
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        B = randn(T, n, n)
        A = Matrix(Hermitian(B * B' + n * I))
        for i in 1:n, j in 1:n
            F = UpdatableCholesky(cholesky(Hermitian(A, uplo)))
            shift_columns!(F, i, j)
            p = UpdatableFactorizations._cyclicperm(n, i, j)
            L = Matrix(F)
            @test norm(L * L' - A[p, p]) / norm(A) < 1e-12
        end
    end
end

@testitem "Cholesky symmetric insertion, every index" begin
    using LinearAlgebra
    n = 6
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        B = randn(T, n + 1, n + 1)
        A = Matrix(Hermitian(B * B' + (n + 1) * I))
        for j in 1:(n + 1)
            keep = [k for k in 1:(n + 1) if k != j]
            F = UpdatableCholesky(cholesky(Hermitian(A[keep, keep], uplo)))
            insert_column!(F, j, A[:, j])
            L = Matrix(F)
            @test size(F) == (n + 1, n + 1)
            @test norm(L * L' - A) / norm(A) < 1e-12
        end
    end
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — `shift_columns!` not defined.

- [ ] **Step 3: Write the implementation**

Append to `src/cholesky_resize.jl`:

```julia
# Restore lower-triangular form of `B` by right multiplication with Givens rotations. Entries
# that are already zero are skipped, so a permutation disturbing only a band costs only that
# band rather than a full re-triangularization.
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

function _permute!(F::UpdatableCholesky, perm::AbstractVector{Int})
    n = F.n
    L = _lower(F)
    B = Matrix(L)[perm, :]
    _lq!(B)
    for c in 1:n, r in 1:n
        L[r, c] = r >= c ? B[r, c] : zero(eltype(B))
    end
    return F
end

# The permutation that moves the index at position `i` to position `j`, sliding the indices
# between them by one.
_cyclicperm(n::Int, i::Int, j::Int) = i <= j ?
    [1:(i - 1); (i + 1):j; i; (j + 1):n] :
    [1:(j - 1); i; j:(i - 1); (i + 1):n]

"""
    shift_columns!(F::UpdatableCholesky, i, j) -> F

Move index `i` to position `j`, sliding the indices between them by one, and update the
factorization to match. Both the row and the column move, keeping the factored matrix symmetric.

Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5.
"""
function shift_columns!(F::UpdatableCholesky, i::Integer, j::Integer)
    1 <= i <= F.n || throw(BoundsError(F, i))
    1 <= j <= F.n || throw(BoundsError(F, j))
    return _permute!(F, _cyclicperm(F.n, Int(i), Int(j)))
end

"""
    insert_column!(F::UpdatableCholesky, j, x) -> F

Insert a new index at position `j`, adding both a row and a column. `x` is the new row and
column in the resulting indexing, so it has length `n+1` and `x[j]` is the new diagonal entry.

Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating the
Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772-795.
"""
function insert_column!(F::UpdatableCholesky, j::Integer, x::AbstractVector)
    n = F.n
    length(x) == n + 1 ||
        throw(DimensionMismatch("x has length $(length(x)), expected $(n + 1)"))
    1 <= j <= n + 1 || throw(BoundsError(F, j))
    _append!(F, vcat(x[1:(j - 1)], x[(j + 1):(n + 1)], x[j]))
    return j == n + 1 ? F : shift_columns!(F, n + 1, Int(j))
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS. The shift test covers all 49 index pairs per element type and orientation.

- [ ] **Step 5: Commit**

```bash
git add src/cholesky_resize.jl test/cholesky_resize.jl
git commit -m "Add Cholesky index shift and insertion"
```

---

### Task 8: `UpdatableLU` type

**Files:**
- Create: `src/lu_type.jl`, `test/lu_type.jl`

**Interfaces:**
- Consumes: nothing
- Produces: `UpdatableLU{T,S}` with fields `Lf::S`, `d::Vector{T}`, `Uf::S`, `p::Vector{Int}`, `pivoted::Bool`, `work::Vector{T}`; `UpdatableLU(::LU)`; properties `L`, `U`, `P`; `size`, `ldiv!`, `\`, `det`

The factorization is stored as `P*A = L*Diagonal(d)*U` with `L` unit lower triangular and `U`
unit upper triangular, because that is the form Bennett's update acts on. `F.L` and `F.U` are
reassembled on property access to match what `LinearAlgebra.lu` returns.

- [ ] **Step 1: Write the failing test**

`test/lu_type.jl`:

```julia
@testitem "UpdatableLU reconstructs its matrix" begin
    using LinearAlgebra
    for T in (Float64, ComplexF64), pivot in (true, false)
        n = 6
        A = randn(T, n, n) + n * I
        G = pivot ? lu(A) : lu(A, NoPivot())
        F = UpdatableLU(G)
        @test norm(F.L * F.U - A[F.p, :]) / norm(A) < 1e-12
        @test size(F) == (n, n)
    end
end

@testitem "UpdatableLU solves" begin
    using LinearAlgebra
    n = 6
    A = randn(n, n) + n * I
    b = randn(n)
    F = UpdatableLU(lu(A))
    @test norm(A * (F \ b) - b) / norm(b) < 1e-11
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — `UpdatableLU` not defined.

- [ ] **Step 3: Write the implementation**

`src/lu_type.jl`:

```julia
"""
    UpdatableLU(G::LU)
    UpdatableLU(A::AbstractMatrix; pivot = RowMaximum())

LU factorization that supports rank-1 update. Stored as `P*A = L*Diagonal(d)*U` with `L` unit
lower triangular and `U` unit upper triangular; `F.L` and `F.U` reassemble the factors in the
form `LinearAlgebra.lu` returns.
"""
mutable struct UpdatableLU{T, S <: AbstractMatrix{T}} <: Factorization{T}
    Lf::S
    d::Vector{T}
    Uf::S
    p::Vector{Int}
    pivoted::Bool
    work::Vector{T}
end

function UpdatableLU(G::LU{T}) where {T}
    n = size(G, 1)
    Lf = Matrix(UnitLowerTriangular(G.factors))
    U = Matrix(UpperTriangular(G.factors))
    d = T[U[k, k] for k in 1:n]
    Uf = Matrix{T}(I, n, n)
    for k in 1:n
        iszero(d[k]) && throw(ZeroPivotException(k))
        for j in (k + 1):n
            Uf[k, j] = U[k, j] / d[k]
        end
    end
    return UpdatableLU{T, Matrix{T}}(Lf, d, Uf, collect(G.p), G.p != 1:n, zeros(T, n))
end

UpdatableLU(A::AbstractMatrix; pivot = RowMaximum()) = UpdatableLU(lu(A, pivot))

Base.size(F::UpdatableLU) = (length(F.d), length(F.d))
Base.size(F::UpdatableLU, i::Integer) = i <= 2 ? length(F.d) : 1

function Base.getproperty(F::UpdatableLU, s::Symbol)
    s === :L && return UnitLowerTriangular(getfield(F, :Lf))
    s === :U && return Diagonal(getfield(F, :d)) * UnitUpperTriangular(getfield(F, :Uf))
    return getfield(F, s)
end

Base.propertynames(::UpdatableLU) = (:L, :U, :p, :pivoted)

function LinearAlgebra.ldiv!(F::UpdatableLU, b::AbstractVector)
    permute!(b, F.p)
    ldiv!(UnitLowerTriangular(getfield(F, :Lf)), b)
    b ./= getfield(F, :d)
    ldiv!(UnitUpperTriangular(getfield(F, :Uf)), b)
    return b
end

Base.:\(F::UpdatableLU, b::AbstractVector) = ldiv!(F, copy(b))

LinearAlgebra.det(F::UpdatableLU) =
    prod(getfield(F, :d)) * (isodd(_permutation_parity(getfield(F, :p))) ? -1 : 1)

function _permutation_parity(p::AbstractVector{Int})
    q = collect(p)
    swaps = 0
    for i in eachindex(q)
        while q[i] != i
            j = q[i]
            q[i], q[j] = q[j], q[i]
            swaps += 1
        end
    end
    return swaps
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lu_type.jl test/lu_type.jl
git commit -m "Add UpdatableLU and its solve"
```

---

### Task 9: LU rank-1 update

**Files:**
- Create: `src/lu_update.jl`, `test/lu_update.jl`

**Interfaces:**
- Consumes: `UpdatableLU`
- Produces: `LinearAlgebra.lowrankupdate!(F::UpdatableLU, u, v)`; internal `_bennett!(L, d, U, w, z, sigma)`

- [ ] **Step 1: Write the failing test**

`test/lu_update.jl`:

```julia
@testitem "LU rank-1 update, pivoted and not" begin
    using LinearAlgebra
    for T in (Float64, ComplexF64), pivot in (true, false)
        n = 7
        A = randn(T, n, n) + n * I
        u = randn(T, n)
        v = randn(T, n)
        F = UpdatableLU(pivot ? lu(A) : lu(A, NoPivot()))
        lowrankupdate!(F, u, v)
        @test norm(F.L * F.U - (A + u * v')[F.p, :]) / norm(A) < 1e-11
    end
end

@testitem "LU rank-1 update does not consume its vectors" begin
    using LinearAlgebra
    n = 6
    A = randn(n, n) + n * I
    u = randn(n)
    v = randn(n)
    ucopy, vcopy = copy(u), copy(v)
    F = UpdatableLU(lu(A))
    lowrankupdate!(F, u, v)
    @test u == ucopy
    @test v == vcopy
end

@testitem "LU rank-1 update rejects mismatched vectors" begin
    using LinearAlgebra
    F = UpdatableLU(lu(randn(6, 6) + 6I))
    @test_throws "u has length 5, factorization is 6" lowrankupdate!(F, randn(5), randn(6))
    @test_throws "v has length 7, factorization is 6" lowrankupdate!(F, randn(6), randn(7))
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — no three-argument `lowrankupdate!` for `UpdatableLU`.

- [ ] **Step 3: Write the implementation**

`src/lu_update.jl`:

```julia
"""
    lowrankupdate!(F::UpdatableLU, u, v) -> F

Replace the factorization of `A` with that of `A + u*v'` in `O(n^2)` operations. Neither `u` nor
`v` is modified. The permutation is retained rather than recomputed, so a pivot that becomes
small is not repaired; `ZeroPivotException` is thrown if one reaches zero.

Bennett, *Triangular factors of modified matrices*, Numerische Mathematik 7 (1965), 217-221.
The pivoted case follows Stange, Griewank and Bollhöfer, *On the efficient update of rectangular
LU-factorizations subject to low rank modifications*, ETNA 26 (2007), 161-177, in applying the
update in the permuted frame.
"""
function LinearAlgebra.lowrankupdate!(F::UpdatableLU{T}, u::AbstractVector,
                                      v::AbstractVector) where {T}
    n = length(getfield(F, :d))
    length(u) == n || throw(DimensionMismatch("u has length $(length(u)), factorization is $n"))
    length(v) == n || throw(DimensionMismatch("v has length $(length(v)), factorization is $n"))
    p = getfield(F, :p)
    w = Vector{T}(undef, n)
    for i in 1:n
        w[i] = u[p[i]]
    end
    # `_bennett!` works in the transpose form, so A + u*v' is passed as second vector conj(v).
    # `conj(v)` alone would alias for real `v` -- Base defines conj(::AbstractArray{<:Real}) = v
    # -- and the kernel consumes what it is given.
    z = Vector{T}(undef, n)
    for i in 1:n
        z[i] = conj(v[i])
    end
    _bennett!(getfield(F, :Lf), getfield(F, :d), getfield(F, :Uf), w, z, one(T))
    return F
end

# A + sigma*w*transpose(z), on the LDU form. `w` and `z` are consumed.
function _bennett!(L, d, U, w, z, sigma)
    n = length(d)
    s = sigma
    for k in 1:n
        wk = w[k]
        zk = z[k]
        dnew = d[k] + s * wk * zk
        iszero(dnew) && throw(ZeroPivotException(k))
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lu_update.jl test/lu_update.jl
git commit -m "Add LU rank-1 update"
```

---

### Task 10: Contracts and StrictMode gates

**Files:**
- Create: `src/contracts.jl`, `test/strict.jl`

**Interfaces:**
- Consumes: both factorization types and all kernels
- Produces: `@invariants` on both types; `@test_typestable` / `@test_noalloc` coverage of the rank-1 kernels

- [ ] **Step 1: Write the failing test**

`test/strict.jl`:

```julia
@testitem "rank-1 kernels are type stable" begin
    using LinearAlgebra, StrictModeTest
    n = 8
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))
    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    v = randn(n)
    @test_typestable lowrankupdate!(F, v)
    @test_typestable lowrankdowndate!(F, v ./ 16)

    G = UpdatableLU(lu(randn(n, n) + n * I))
    @test_typestable lowrankupdate!(G, randn(n), randn(n))
end

@testitem "Cholesky rank-1 update allocates nothing" begin
    using LinearAlgebra, StrictModeTest
    n = 8
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))
    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    v = randn(n)
    @test_noalloc lowrankupdate!(F, v)
end

@testitem "factorization invariants hold after every operation" begin
    using LinearAlgebra, TypeContracts
    n = 7
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))
    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    lowrankupdate!(F, randn(n))
    @test check_invariants(F)
    delete_column!(F, 3)
    @test check_invariants(F)
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — `check_invariants` not defined for these types; `@test_noalloc` fails because
`lowrankdowndate!` allocates `cs` and `sn`.

- [ ] **Step 3: Write the implementation**

Two changes.

First, `src/contracts.jl`:

```julia
using TypeContracts

@invariants UpdatableCholesky begin
    F.n >= 0
    F.n <= size(F.factors, 1)
    F.uplo == 'L' || F.uplo == 'U'
    length(F.work) >= F.n
end

@invariants UpdatableLU begin
    length(F.d) == size(F.Lf, 1)
    length(F.p) == length(F.d)
    sort(F.p) == 1:length(F.p)
end
```

Second, make `_ch1dn!` allocation-free by taking its two rotation buffers from the
factorization's workspace. Change `UpdatableCholesky` to carry `rot::Vector{T}` and
`cosines::Vector{real(T)}` alongside `work`, sized to `capacity`, and change `_ch1dn!` to accept
them:

```julia
function LinearAlgebra.lowrankdowndate!(F::UpdatableCholesky, v::AbstractVector)
    length(v) == F.n ||
        throw(DimensionMismatch("v has length $(length(v)), factorization is $(F.n)"))
    w = view(F.work, 1:F.n)
    copyto!(w, v)
    _ch1dn!(_lower(F), w, view(F.cosines, 1:F.n), view(F.rot, 1:F.n))
    return F
end
```

with `_ch1dn!(L, w, cs, sn)` using the passed buffers instead of allocating them. Update
`_grow!` to resize all three buffers.

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS. If `@test_noalloc` still reports allocations on `lowrankupdate!`, run
`StrictMode.@assert_noalloc` interactively and read `kernel_report` to find the source before
changing anything.

- [ ] **Step 5: Commit**

```bash
git add src/contracts.jl src/cholesky_type.jl src/cholesky_update.jl test/strict.jl
git commit -m "Add invariants and allocation-free rank-1 kernels"
```

---

### Task 11: Generic indexing and element types

**Files:**
- Create: `test/generic.jl`

**Interfaces:**
- Consumes: everything
- Produces: no new API; a regression net over unusual inputs

- [ ] **Step 1: Write the failing test**

`test/generic.jl`:

```julia
@testitem "non-BLAS element types" begin
    using LinearAlgebra, ForwardDiff
    n = 5
    for T in (Float32, BigFloat, ForwardDiff.Dual{Nothing, Float64, 1})
        B = T.(randn(n, n))
        A = Matrix(Symmetric(B * B' + n * I))
        v = T.(randn(n))
        F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
        lowrankupdate!(F, v)
        L = Matrix(F)
        @test norm(L * L' - (A + v * v')) / norm(A) < sqrt(eps(Float64))
    end
end

@testitem "offset and viewed inputs" begin
    using LinearAlgebra, OffsetArrays
    n = 5
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))
    v = randn(n)
    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    G = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    lowrankupdate!(F, v)
    lowrankupdate!(G, OffsetVector(v, 0:(n - 1)))
    @test Matrix(F) ≈ Matrix(G)

    H = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    lowrankupdate!(H, view(vcat(v, randn(3)), 1:n))
    @test Matrix(F) ≈ Matrix(H)
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: the `OffsetVector` case fails, because `copyto!(w, v)` between differently-axed
vectors throws or misaligns.

- [ ] **Step 3: Write the implementation**

In every entry point that copies a user vector into the workspace, copy by position rather than
by index so an offset input is accepted:

```julia
w = view(F.work, 1:F.n)
for (i, x) in enumerate(v)
    w[i] = x
end
```

Apply the same in `lowrankupdate!` for `UpdatableLU`, where `u[p[i]]` must become
`u[first(eachindex(u)) + p[i] - 1]`, or more simply operate on `collect`ed input when
`axes(u, 1) != 1:n`. Prefer the explicit check:

```julia
Base.require_one_based_indexing(getfield(F, :Lf))
```

on the stored factors, while accepting arbitrary axes on the input vectors.

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS. `BigFloat` and `Dual` exercise the generic path; `Float32` stays on BLAS types.

- [ ] **Step 5: Commit**

```bash
git add test/generic.jl src/
git commit -m "Accept offset input vectors and non-BLAS element types"
```

---

### Task 12: Documentation with provenance

**Files:**
- Create: `docs/Project.toml`, `docs/make.jl`, `docs/src/index.md`, `docs/src/getting_started.md`, `docs/src/updating.md`, `docs/src/provenance.md`, `docs/src/api.md`, `.github/workflows/Documentation.yml`

**Interfaces:**
- Consumes: the whole public API
- Produces: a deployed DocumenterVitepress site

- [ ] **Step 1: Write `docs/make.jl`**

Use `DocumenterVitepress.deploydocs`, not `Documenter.deploydocs` — the latter produces a 404.

```julia
using Documenter, DocumenterVitepress, UpdatableFactorizations

makedocs(;
    sitename = "UpdatableFactorizations.jl",
    modules = [UpdatableFactorizations],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/el-oso/UpdatableFactorizations.jl",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Updating and downdating" => "updating.md",
        "Provenance" => "provenance.md",
        "API" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/UpdatableFactorizations.jl",
    push_preview = true,
)
```

- [ ] **Step 2: Write `docs/src/provenance.md`**

Two tables, exactly as the spec requires. The first maps routine to article:

| routine | article |
| --- | --- |
| Cholesky rank-1 update | Gill, Golub, Murray and Saunders, Math. Comp. 28 (1974), 505-535 |
| Cholesky rank-1 downdate | Bojanczyk, Brent, Van Dooren and de Hoog, SISSC 8 (1987), 210-221 |
| Cholesky symmetric delete and shift | Golub and Van Loan, *Matrix Computations*, 4th ed., section 6.5 |
| Cholesky symmetric insert | Daniel, Gragg, Kaufman and Stewart, Math. Comp. 30 (1976), 772-795 |
| LU rank-1 update, unpivoted | Bennett, Numer. Math. 7 (1965), 217-221 |
| LU rank-1 update, pivoted | Stange, Griewank and Bollhöfer, ETNA 26 (2007), 161-177 |

The second lists reference implementations consulted:

| implementation | license | consulted for |
| --- | --- | --- |
| `LinearAlgebra` (Julia stdlib) | MIT | the `lowrankupdate!`/`lowrankdowndate!` signatures this package extends |
| `QRupdate.jl` | MIT | API shape |
| `UpdatableQRFactorizations.jl` | MIT | API shape |
| `UpdatableCholeskyFactorizations.jl` | MIT | the capacity-with-active-block storage idea |

Followed by, verbatim in substance: `qrupdate-ng` is GPL-3.0-or-later; its source was not read.
It is used only as a benchmark target and as the reference list of routine names.
`QRupdatesFast.jl` links `qrupdate-ng` and so appears only in the benchmark environment.

- [ ] **Step 3: Write the remaining pages**

`index.md` states what the package does and that Cholesky rank-1 update and downdate extend the
standard library's functions rather than replacing them. `updating.md` gives a worked example for
each of the six operations shipped in this milestone. `api.md` is `@autodocs` over the module.
No page claims the package is faster than LAPACK.

- [ ] **Step 4: Build the docs locally**

Run `julia --project=docs docs/make.jl` once. Expected: builds with no missing-docstring warnings.

- [ ] **Step 5: Commit**

```bash
git add docs .github/workflows/Documentation.yml
git commit -m "Add documentation with a provenance page"
```

---

### Task 13: Full-suite gate

**Files:**
- Modify: none expected

- [ ] **Step 1: Run the whole suite**

`julia_run_testitems` over the package with `max_workers = 4`, no filter.
Expected: every item passes.

- [ ] **Step 2: Run `Pkg.test()` once, cold**

This is the pre-commit gate, and the only place a cold Julia process is warranted.
Expected: passes, including the `StrictModeTest` proofs.

- [ ] **Step 3: Format**

Run `runic` over `src/` and `test/`. Commit any reformatting separately.

- [ ] **Step 4: Check the diff's comments**

Re-read every comment added by this milestone. Any comment that references this plan, a
milestone, a task number, or how the code came to be is a defect; rewrite it to state what is
true of the code now.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Format and finish Cholesky and LU updating"
```

---

## Self-review

**Spec coverage for this milestone.** R1 (Task 1 and 12), R7 for the Cholesky and LU half
(Tasks 3-7, 9), R9 (Task 10), R10 (Task 10), R11 (Task 11), R12 and R15 (Task 12). R2-R6 and R14
belong to M1, R8 to M3, R13 to M4 — out of scope here and covered by their own plans.

**Deferred to M2b, deliberately.** `UpdatableQR`, `AbstractQRep`, `DenseQ`, and the QR verbs.
The spec places the Layer-1 types in M0; splitting `UpdatableQR` into M2b keeps this plan's
deliverable self-contained, since nothing in M2a uses a Q representation.

**Known gap carried forward.** `insert_column!` allocates through `vcat` and `_permute!`
allocates a dense copy in `Matrix(L)[perm, :]`. Both are `O(n^2)` operations where an `O(n^2)`
allocation is not a complexity regression, and neither carries a `@test_noalloc` assertion.
Making them allocation-free is an M4 optimization, not a correctness matter.

**Accuracy note for the docs.** Bennett's update measured a median relative error of 1e-16 to
1e-15 on well-conditioned diagonally dominant matrices at n=50, with the worst of 800 trials at
2.4e-13 when the update norm reached 100 times the matrix norm. Its known weakness is
near-breakdown pivots, which those trials do not reach; the `ZeroPivotException` guard in Task 9
catches only exact breakdown. `updating.md` should say this plainly rather than claim backward
stability.
