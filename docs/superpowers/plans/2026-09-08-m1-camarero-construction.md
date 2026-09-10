# M1: Camarero Algorithms 1-3, the construction layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `cholesky_crout`, `lu_crout` (unpivoted and partially pivoted) and `qr_bcgs` (rectangular `m >= n`, optional reorthogonalization) — the three blocked left-looking factorizations of Camarero, arXiv:1812.02056 — with the deferred flush exposed as plain `rankk!` and `matmul!` keyword arguments, and settle locally the two ratios the spike never measured.

**Architecture:** Each routine is a column-at-a-time Crout/block-CGS loop that defers the trailing update and flushes it every `s` columns through one substitutable function. There is no abstract type over the flush, no extension, and no plugin registry: `rankk!` and `matmul!` are function-valued keyword arguments whose defaults are `default_rankk!` (BLAS `syrk!`/`herk!`, or a generic `mul!` when the element type is not a BLAS float) and `LinearAlgebra.mul!`. `cholesky_crout` and `lu_crout` hand their finished factors to the standard library's `Cholesky` and `LU` types and let the shipped `UpdatableCholesky(::Cholesky)` and `UpdatableLU(::LU)` constructors do the normalization, so no new struct and no new conversion path is introduced. `qr_bcgs` hands its thin `Q` and its `R` to the shipped `UpdatableQR(Q, R; capacity)` constructor and does the same.

**Tech Stack:** Julia 1.12, `LinearAlgebra`, `TypeContracts` 0.14; `TestItems`/`TestItemRunner`, `StrictModeTest`, `ForwardDiff`, `OffsetArrays` (test only); `Chairmarks` and `JSON` (bench only); DocumenterVitepress.

**Spec:** `docs/superpowers/specs/2026-09-08-updatablefactorizations-design.md`, sections 3, 4 (Layer 2), 5, 6, 7 and requirements R2-R6 and R14.

**Verified reference implementation:** `docs/superpowers/specs/alg1.jl` and `alg23.jl`, with the raw sweep in `RESULTS.md`. The kernels in those two files were run against freshly recomputed factorizations before this plan was written; every relative residual was 1.3e-16 to 2.8e-16. The code blocks below are those kernels, renamed, generalized to complex and non-BLAS element types, given the substitutable flush and given fail-fast guards. Do not restructure a loop without re-checking it against a recomputed factorization for both element types and a block size that is not a divisor of `n`.

## The honest performance statement

These algorithms are **slower than blocked LAPACK** and this plan does not try to change that. From `RESULTS.md`, single-threaded on an unpinned-clock host, ratios against the LAPACK baseline (higher is better; above 1.00 would be a win):

| algorithm | best flush | baseline | n=2000 | n=4000 |
| --- | --- | --- | --- | --- |
| Alg 1, Cholesky | `syrk`, lower triangle only | `potrf` | 0.88 | 0.93 |
| Alg 1, Cholesky | paper-faithful full-block `gemm` | `potrf` | 0.58 | 0.63 |
| Alg 2, LU, unpivoted | `gemm` | `getrf`, pivoted | 0.95 | 0.91 |
| Alg 3, QR | `gemm` | `geqrf` + forming thin Q | 0.92 | 0.93 |

Strassen makes Algorithm 1 **worse**, not better (0.63 to 0.47 at n=4000, s=256): the flush is a rank-`s` update whose inner dimension is `s`, so the flop cut cannot pay for the buffer traffic. Algorithm 3 additionally gives up roughly 50x in orthogonality (`norm(Q'Q - I)` of 1.2e-12 to 7.2e-12 against Householder's 6.8e-14 to 1.0e-13) and 20-60x in `norm(QR - A)/norm(A)`.

The one measured win is unpivoted LU: `lu!(A, NoPivot())` in the standard library is an unblocked generic fallback running at 0.05-0.06 of pivoted `getrf`, and Algorithm 2 beats it by 14.7x at n=2000 and 17.8x at n=4000. That is the sentence the documentation is allowed to make, and it is a statement about a gap in the standard library, not about beating LAPACK.

Every task in this plan, and every document it produces, states this plainly. A task that produces prose implying otherwise has failed.

## Prerequisites

M1 runs after M2b. The construction routines return Layer-1 types, and `qr_bcgs` returns an
`UpdatableQR`, so the QR updating layer is on the branch before this milestone starts. Two names
from it are consumed here:

```julia
UpdatableQR(Q::AbstractMatrix, R::AbstractMatrix; capacity = (2size(Q, 1), 2size(Q, 2)))
qr_householder(A::AbstractMatrix; capacity = (2size(A, 1), 2size(A, 2)))
```

The module file already carries the QR layer's exports and `public` names; every edit to it here
adds lines rather than replacing the block.

## Global Constraints

- License MIT. No GPL code in the dependency graph outside `bench/`. `qrupdate-ng` source is never read.
- `julia = "1.12"`. `TypeContracts = "0.14"`. `StrictMode` is not a dependency and this milestone does not add it (see Task 8 and the Self-review).
- No `@inbounds` anywhere in this plan.
- Comments, docstrings and commit messages state what is true of the code now. No reference to this plan, to a milestone, a task number, a review, or how the code came to be.
- Every algorithm's docstring cites its source article by author, title, journal, volume, year and pages, in the format already used at `src/cholesky_resize.jl:11` and `src/lu_update.jl:16-19`.
- No document claims a speed win over LAPACK. The two claims flagged unmeasured in spec section 3 — that blocking helps non-BLAS element types, and that pivoted `lu_crout` retains the unpivoted ratio — are not stated as fact anywhere until Task 10 measures them.
- Run tests with the JuliaMCP `julia_run_testitems` tool, always passing `max_workers`. Shell out to `Pkg.test()` only as the final cold pre-commit gate (Task 12).
- `Project.toml` and `bench/Project.toml` are edited only through `Pkg` (`Pkg.add`, `Pkg.compat`, `Pkg.develop`) from an MCP Julia session. Never hand-write a UUID.
- Pin `BLAS.set_num_threads(1)` at the top of every timing script. A benchmark process inherits whatever the previous script set.
- American spellings.

## File Structure

| file | responsibility |
| --- | --- |
| `src/UpdatableFactorizations.jl` | module: gains `include("construct.jl")`, three exports, one `public` name |
| `src/construct.jl` | `default_rankk!`, `cholesky_crout`, `lu_crout`, `qr_bcgs` |
| `test/construct.jl` | one `@testitem` per routine and per failure path |
| `test/generic.jl` | gains the three routines in the non-BLAS eltype loop and the seeded-partial block |
| `test/strict.jl` | gains the return-type concreteness item |
| `bench/Project.toml` | `Chairmarks`, `JSON`, `ForwardDiff`, `LinearAlgebra`, `Random`, `Printf`, `Dates`, the package by path |
| `bench/construct_sweep.jl` | the R14 sweep; writes every sample to `bench/results/` |
| `bench/results/construction-<host>.json` | saved datapoints, committed |
| `docs/src/construction.md` | the construction page, carrying the ratio table above |
| `docs/src/provenance.md` | gains the Camarero and Giraud/Langou/Rozložník rows |
| `docs/make.jl` | gains the construction page |

`docs/src/api.md` is not edited: it is an `@autodocs` block over the whole module, which picks up
every newly documented name, exported or `public`, on its own.

---

### Task 1: Construction file, exports, and the default rank-k flush

**Files:**
- Create: `src/construct.jl`, `test/construct.jl`
- Modify: `src/UpdatableFactorizations.jl`

**Interfaces:**
- Consumes: `LinearAlgebra.mul!`, `LinearAlgebra.BLAS.syrk!`, `LinearAlgebra.BLAS.herk!`, `LinearAlgebra.BlasReal`, `LinearAlgebra.BlasComplex`
- Produces: `default_rankk!(C, A, alpha, beta; uplo::Char = 'L') -> C`

- [ ] **Step 1: Write the failing test**

Create `test/construct.jl`:

```julia
@testitem "default_rankk! computes a symmetric rank-k update" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    for T in (Float64, Float32, ComplexF64, BigFloat)
        A = randn(T, 9, 4)
        C0 = Matrix(Hermitian(randn(T, 9, 9)))
        want = -A * A' + C0
        C = copy(C0)
        UpdatableFactorizations.default_rankk!(C, A, -one(T), one(T); uplo = 'L')
        # The BLAS path writes the lower triangle only, so only that triangle is compared.
        @test norm(tril(C) - tril(want)) / norm(want) < 100 * eps(real(T))
    end
end

@testitem "default_rankk! leaves the untouched triangle alone on BLAS element types" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    A = randn(9, 4)
    C = zeros(9, 9)
    C[1, 9] = 17.0
    UpdatableFactorizations.default_rankk!(C, A, -1.0, 1.0; uplo = 'L')
    @test C[1, 9] == 17.0
end
```

- [ ] **Step 2: Run the tests to verify they fail**

`julia_run_testitems` with `max_workers = 4`, filtered to `default_rankk!`.
Expected: FAIL — `UpdatableFactorizations.default_rankk!` is undefined.

- [ ] **Step 3: Write the implementation**

Create `src/construct.jl`:

```julia
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
    R = real(T)
    return BLAS.herk!(uplo, 'N', R(real(alpha)), A, R(real(beta)), C)
end

default_rankk!(C, A, alpha, beta; uplo::Char = 'L') = mul!(C, A, A', alpha, beta)
```

Then in `src/UpdatableFactorizations.jl`, add the include after `lu_update.jl` and before
`contracts.jl`, and add two lines to the declarations already there. The module carries the QR
updating layer's exports and `public` names; leave them as they are and append:

```julia
export cholesky_crout, lu_crout, qr_bcgs
public default_rankk!
```

```julia
include("construct.jl")
```

The export block afterwards reads:

```julia
export UpdatableCholesky, UpdatableLU, UpdatableQR
export insert_column!, delete_column!, shift_columns!, insert_row!, delete_row!
export qr_householder
export cholesky_crout, lu_crout, qr_bcgs
public AbstractQRep, DenseQ, materialize, capacity
public default_rankk!
```

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS, both items. The `BigFloat` cell exercises the generic method; the `Float32`,
`Float64` and `ComplexF64` cells exercise the two BLAS methods.

- [ ] **Step 5: Commit**

```bash
git add src/construct.jl src/UpdatableFactorizations.jl test/construct.jl
git commit -m "Add the default symmetric rank-k flush"
```

---

### Task 2: `cholesky_crout`

**Files:**
- Modify: `src/construct.jl`, `test/construct.jl`

**Interfaces:**
- Consumes: `default_rankk!`, `UpdatableCholesky(::Cholesky; capacity)`, `LinearAlgebra.PosDefException`, `LinearAlgebra.checksquare`
- Produces: `cholesky_crout(A::AbstractMatrix{T}; s::Int = 64, uplo::Symbol = :L, capacity::Int = 2size(A, 1), rankk! = default_rankk!) -> UpdatableCholesky{T, real(T), Matrix{T}}`

- [ ] **Step 1: Write the failing test**

Append to `test/construct.jl`:

```julia
@testitem "cholesky_crout factors" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 9
    for uplo in (:L, :U), T in (Float64, ComplexF64), s in (1, 4, 64)
        B = randn(T, n, n)
        A = Matrix(Hermitian(B * B' + n * I))
        # The triangle `uplo` does not name is overwritten with garbage, so an implementation
        # that reads the wrong triangle, or ignores the keyword, cannot pass both halves of the
        # loop. `A` is the Hermitian completion of the triangle that is read.
        G = copy(A)
        for j in 1:n, i in 1:(j - 1)
            uplo === :L ? (G[i, j] = 1.0e6 * (i + j)) : (G[j, i] = 1.0e6 * (i + j))
        end
        F = cholesky_crout(G; s, uplo)
        L = Matrix(F.L)
        @test norm(L * L' - A) / norm(A) < 1.0e-13
        @test all(x -> abs(imag(x)) < 1.0e-12 && real(x) > 0, diag(L))
        @test size(F) == (n, n)
        @test UpdatableFactorizations.capacity(F) == 2n
        @test norm(A * (F \ ones(T, n)) - ones(T, n)) < 1.0e-11
    end
end

@testitem "cholesky_crout honors capacity and keeps updating available" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 6
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))
    F = cholesky_crout(A; s = 2, capacity = 20)
    @test UpdatableFactorizations.capacity(F) == 20
    v = randn(n)
    lowrankupdate!(F, v)
    @test norm(Matrix(F) - (A + v * v')) / norm(A) < 1.0e-13
end

@testitem "cholesky_crout throws on a matrix that is not positive definite" begin
    using LinearAlgebra
    A = Matrix(Symmetric([1.0 2.0 0.0; 2.0 1.0 0.0; 0.0 0.0 1.0]))
    # The second pivot is the first non-positive one, and the exception names it.
    @test_throws PosDefException(2) cholesky_crout(A; s = 2)
end

@testitem "cholesky_crout rejects a block size below one" begin
    using LinearAlgebra
    A = Matrix(1.0I, 4, 4)
    @test_throws "block size s must be at least 1, got 0" cholesky_crout(A; s = 0)
end

@testitem "cholesky_crout rejects an offset matrix" begin
    using LinearAlgebra, OffsetArrays
    A = OffsetMatrix(Matrix(1.0I, 4, 4), 0:3, 0:3)
    @test_throws "offset arrays are not supported" cholesky_crout(A; s = 2)
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — `cholesky_crout` is undefined.

- [ ] **Step 3: Write the implementation**

Append to `src/construct.jl`:

```julia
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
    M = Matrix(Hermitian(A, uplo))
    L = zeros(T, n, n)
    panel = Vector{T}(undef, n)
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
    return UpdatableCholesky(Cholesky(L, 'L', 0); capacity)
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS, all five items. If the `s = 1` cell fails while `s = 64` passes, the flush
boundary is wrong: at `c == z + s` the flush must remove the contribution of columns `z` through
`c-1` only, and `z` is reassigned to `c` immediately after.

- [ ] **Step 5: Commit**

```bash
git add src/construct.jl test/construct.jl
git commit -m "Add blocked Crout Cholesky construction"
```

---

### Task 3: `lu_crout`, unpivoted

**Files:**
- Modify: `src/construct.jl`, `test/construct.jl`

**Interfaces:**
- Consumes: `LinearAlgebra.mul!`, `LinearAlgebra.NoPivot`, `LinearAlgebra.ZeroPivotException`, `UpdatableLU(::LU)`
- Produces: `lu_crout(A::AbstractMatrix{T}; s::Int = 64, pivot = RowMaximum(), rtol::Real = 0, matmul! = mul!) -> UpdatableLU{T, Matrix{T}}`; internal `_pivotrow(pivot, col) -> Int`

- [ ] **Step 1: Write the failing test**

Append to `test/construct.jl`:

```julia
@testitem "lu_crout without pivoting factors" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 9
    for T in (Float64, ComplexF64), s in (1, 4, 64)
        A = randn(T, n, n) + n * I
        F = lu_crout(A; s, pivot = NoPivot())
        # With no pivoting the permutation must be the identity, which is what makes the
        # residual below a test of the factorization rather than of the permutation.
        @test F.p == 1:n
        @test norm(F.L * F.U - A) / norm(A) < 1.0e-12
        @test norm(A * (F \ ones(T, n)) - ones(T, n)) < 1.0e-10
    end
end

@testitem "lu_crout without pivoting throws on a zero pivot" begin
    using LinearAlgebra
    A = [1.0 2.0 3.0; 2.0 4.0 5.0; 1.0 1.0 1.0]   # the leading 2x2 minor is singular
    # The second pivot is the one that vanishes, and the exception names it.
    @test_throws ZeroPivotException(2) lu_crout(A; s = 2, pivot = NoPivot())
end

@testitem "lu_crout rtol widens the zero-pivot test" begin
    using LinearAlgebra
    A = [1.0 2.0 3.0; 2.0 (4.0 + 1.0e-13) 5.0; 1.0 1.0 1.0]
    F = lu_crout(A; s = 2, pivot = NoPivot())
    @test issuccess(F)
    @test abs(F.d[2]) < 1.0e-12
    @test_throws ZeroPivotException(2) lu_crout(A; s = 2, pivot = NoPivot(), rtol = 1.0e-8)
end

@testitem "lu_crout rejects an unsupported pivoting strategy" begin
    using LinearAlgebra
    A = Matrix(1.0I, 4, 4)
    @test_throws "use NoPivot() or RowMaximum()" lu_crout(A; s = 2, pivot = ColumnNorm())
end

@testitem "lu_crout rejects an offset matrix" begin
    using LinearAlgebra, OffsetArrays
    A = OffsetMatrix(Matrix(1.0I, 4, 4), 0:3, 0:3)
    @test_throws "offset arrays are not supported" lu_crout(A; s = 2)
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — `lu_crout` is undefined.

- [ ] **Step 3: Write the implementation**

Append to `src/construct.jl`:

```julia
# Index within `col` of the row to move into the pivot position.
_pivotrow(::NoPivot, col) = 1
_pivotrow(::RowMaximum, col) = findmax(abs, col)[2]
_pivotrow(pivot, col) = throw(
    ArgumentError("pivoting strategy $pivot is not supported; use NoPivot() or RowMaximum()")
)

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
    U = Matrix{T}(I, n, n)
    ipiv = collect(1:n)
    z = 1
    @views for c in 1:n
        if c == z + s
            matmul!(M[c:n, c:n], L[c:n, z:(c - 1)], U[z:(c - 1), c:n], -one(T), one(T))
            z = c
        end
        col = L[c:n, c]
        copyto!(col, M[c:n, c])
        c > z && matmul!(col, L[c:n, z:(c - 1)], U[z:(c - 1), c], -one(T), one(T))
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
            ipiv[c] = pv
        end
        Lcc = L[c, c]
        (iszero(Lcc) || abs(Lcc) <= rtol * norm(col)) && throw(ZeroPivotException(c))
        if c < n
            row = U[c, (c + 1):n]
            copyto!(row, M[c, (c + 1):n])
            c > z &&
                matmul!(row, transpose(U[z:(c - 1), (c + 1):n]), L[c, z:(c - 1)], -one(T), one(T))
            row ./= Lcc
        end
    end
    # The standard library packs L and U into one array with a unit-diagonal L, which is the
    # form UpdatableLU splits apart again.
    f = Matrix{T}(undef, n, n)
    for j in 1:n
        for i in 1:(j - 1)
            f[i, j] = L[i, i] * U[i, j]
        end
        f[j, j] = L[j, j]
        for i in (j + 1):n
            f[i, j] = L[i, j] / L[j, j]
        end
    end
    return UpdatableLU(LU{T}(f, ipiv, 0))
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS, all five items. The pivoting branch is exercised here only through
`_pivotrow(::NoPivot, col) == 1`, which never swaps; Task 4 covers the swap.

- [ ] **Step 5: Commit**

```bash
git add src/construct.jl test/construct.jl
git commit -m "Add blocked Crout LU construction"
```

---

### Task 4: `lu_crout` with partial pivoting

**Files:**
- Modify: `test/construct.jl`, `src/construct.jl` (only if Step 2 fails)

**Interfaces:**
- Consumes: `lu_crout(A; s, pivot = RowMaximum())`
- Produces: nothing new; this task proves the pivoting path written in Task 3 is correct and is actually reached

The row-interchange protocol is: after column `c` of `L` is fully formed, pick the row of
maximum magnitude over `L[c:n, c]`, interchange it with row `c` in `L[:, 1:c]`, in the
not-yet-consumed columns `M[:, c+1:n]` and in the interchange record. Rows of `U` are untouched,
so the deferred flush stays a rank-`s` product.

- [ ] **Step 1: Write the failing test**

Append to `test/construct.jl`:

```julia
@testitem "lu_crout with partial pivoting factors" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 9
    for T in (Float64, ComplexF64), s in (1, 4, 64)
        A = randn(T, n, n)
        F = lu_crout(A; s)
        # A plain random matrix must produce a nontrivial permutation, or the residual below
        # would pass for an implementation that ignores the pivot entirely.
        @test F.p != 1:n
        @test norm(F.L * F.U - A[F.p, :]) / norm(A) < 1.0e-12
        @test norm(A * (F \ ones(T, n)) - ones(T, n)) < 1.0e-10
    end
end

@testitem "lu_crout with partial pivoting matches the standard library" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 12
    A = randn(n, n)
    F = lu_crout(A; s = 4)
    G = lu(A)
    # Partial pivoting picks the same rows in the same order, so the permutations agree exactly.
    @test F.p == G.p
    @test norm(Matrix(F.L) - Matrix(G.L)) / norm(Matrix(G.L)) < 1.0e-12
    @test norm(Matrix(F.U) - Matrix(G.U)) / norm(Matrix(G.U)) < 1.0e-12
end

@testitem "lu_crout survives a matrix whose unpivoted factorization does not exist" begin
    using LinearAlgebra
    A = [0.0 1.0; 1.0 0.0]
    @test_throws ZeroPivotException(1) lu_crout(A; s = 1, pivot = NoPivot())
    F = lu_crout(A; s = 1)
    @test F.p == [2, 1]
    @test norm(F.L * F.U - A[F.p, :]) < 1.0e-14
end
```

- [ ] **Step 2: Run the tests to verify they fail or pass**

Expected: PASS if Task 3's pivoting branch is correct. If the second item fails on `F.p == G.p`
while the first passes, the interchange record is being written in permutation form rather than
in the standard library's interchange form: `ipiv[c]` is the row swapped with row `c` at step
`c`, not the final destination of row `c`.

- [ ] **Step 3: Fix any failure in `src/construct.jl`**

The one diagnostic: if `F.p == G.p` fails while the residual passes, the interchange record is
being written in permutation form rather than in interchange form. `ipiv[c]` must be the row
swapped with row `c` at step `c`, which is what `UpdatableLU` and the standard library read.

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS, all three items.

- [ ] **Step 5: Commit**

```bash
git add src/construct.jl test/construct.jl
git commit -m "Cover partial pivoting in Crout LU"
```

---

### Task 5: `qr_bcgs`

**Files:**
- Modify: `src/construct.jl`, `test/construct.jl`

**Interfaces:**
- Consumes: `LinearAlgebra.mul!`, `LinearAlgebra.axpy!`, `LinearAlgebra.dot`, `LinearAlgebra.norm`, `UpdatableQR(::AbstractMatrix, ::AbstractMatrix; capacity)`
- Produces: `qr_bcgs(A::AbstractMatrix{T}; s::Int = 64, reorth::Bool = true, rtol::Real = 0, capacity = (2m, 2n), matmul! = mul!) -> UpdatableQR{T}`

- [ ] **Step 1: Write the failing test**

Append to `test/construct.jl`:

```julia
@testitem "qr_bcgs factors a rectangular matrix" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity
    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((12, 7), (9, 9)), s in (1, 3, 64)
        A = randn(T, m, n)
        F = qr_bcgs(A; s)
        @test F isa UpdatableQR{T}
        @test size(F) == (m, n)
        @test size(F.Q) == (m, n)
        @test norm(Matrix(F) - A) / norm(A) < 1.0e-12
        @test norm(F.Q' * F.Q - I) < 1.0e-12
        # `F.R` wraps the stored block in `UpperTriangular`, which reports a zero subdiagonal
        # whatever the block holds, so triangularity is asserted on the block itself.
        Rs = getfield(F, :factors)
        @test all(iszero, [Rs[i, j] for j in 1:n for i in (j + 1):n])
    end
    # The factorization is updatable on return, which is what the Layer-1 return type is for.
    A = randn(12, 7)
    F = qr_bcgs(A; s = 4)
    x = randn(12)
    insert_column!(F, 8, x)
    @test norm(Matrix(F) - [A x]) / norm([A x]) < 1.0e-12
    @test capacity(qr_bcgs(A; s = 4, capacity = (12, 7))) == (12, 7)
end

@testitem "qr_bcgs reorthogonalization recovers orthogonality" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 12
    # A Vandermonde matrix of condition number 8.8e8: block CGS alone loses orthogonality as the
    # square of the condition number, reaching 2.3e-4 here, and one reorthogonalization pass
    # brings it back to 2.3e-12 because cond(A) * eps is still far below one, which is the
    # regime the bound requires.
    A = [x^(j - 1) for x in range(0.0, 1.0; length = n), j in 1:n]
    Qn = qr_bcgs(A; s = 8, reorth = false).Q
    Qy = qr_bcgs(A; s = 8, reorth = true).Q
    @test norm(Qn' * Qn - I) > 1.0e-6
    @test norm(Qy' * Qy - I) < 1.0e-8
end

@testitem "qr_bcgs rejects a wide matrix" begin
    using LinearAlgebra
    @test_throws "is 3 by 5; qr_bcgs requires m >= n" qr_bcgs(randn(3, 5); s = 2)
end

@testitem "qr_bcgs rejects a rank-deficient column" begin
    using LinearAlgebra
    A = [1.0 0.0 1.0; 0.0 1.0 0.0; 0.0 0.0 0.0]
    @test_throws "column 3 is a combination of the columns before it" qr_bcgs(A; s = 2)
    # A column that is dependent only up to rounding leaves noise rather than an exact zero, so
    # the default exact-zero test accepts it and `rtol` is what catches it.
    B = [1.0 0.0 1.0; 0.0 1.0 1.0e-15; 0.0 0.0 1.0e-17]
    @test size(qr_bcgs(B; s = 2)) == (3, 3)
    @test_throws "column 3 is a combination of the columns before it" qr_bcgs(B; s = 2, rtol = 1.0e-8)
end

@testitem "qr_bcgs rejects an offset matrix" begin
    using LinearAlgebra, OffsetArrays
    A = OffsetMatrix(randn(4, 3), 0:3, 0:2)
    @test_throws "offset arrays are not supported" qr_bcgs(A; s = 2)
end
```

- [ ] **Step 2: Run the tests to verify they fail**

Expected: FAIL — `qr_bcgs` is undefined.

- [ ] **Step 3: Write the implementation**

Append to `src/construct.jl`:

```julia
"""
    qr_bcgs(A; s = 64, reorth = true, rtol = 0, capacity = (2m, 2n), matmul! = mul!) -> UpdatableQR

Factor the `m` by `n` matrix `A` with `m >= n` as `Q*R` by block classical Gram-Schmidt and
return it as an [`UpdatableQR`](@ref) holding the thin `Q` (`m` by `n`, orthonormal columns) and
the upper triangular `R`, ready for the updating verbs. Columns are orthogonalized one at a time
against the current block and the accumulated projection against earlier blocks is deferred,
then flushed every `s` columns through `matmul!(C, A, B, alpha, beta)`, which defaults to
`LinearAlgebra.mul!`. `capacity` is passed through to the returned factorization.

`reorth = true` runs the projection a second time at every flush and at every column, which
costs roughly twice as much and gives orthogonality of order `eps` provided the product of the
condition number and `eps` is well below one. `reorth = false` leaves the loss of orthogonality
growing as the square of the condition number. [`qr_householder`](@ref) is more accurate than
either and is the recommended path; see the construction page of the documentation for the
measured ratios and residuals.

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
    M = Matrix{T}(undef, m, n)
    copyto!(M, A)
    Q = zeros(T, m, n)
    coef = Matrix{T}(undef, min(s, n), n)
    z = 1
    @views for c in 1:n
        if c == z + s
            BQ = Q[:, z:(c - 1)]
            BM = M[:, c:n]
            C = coef[1:s, 1:(n - c + 1)]
            matmul!(C, BQ', BM, one(T), zero(T))
            matmul!(BM, BQ, C, -one(T), one(T))
            if reorth
                matmul!(C, BQ', BM, one(T), zero(T))
                matmul!(BM, BQ, C, -one(T), one(T))
            end
            z = c
        end
        v = M[:, c]
        for _ in 1:(reorth ? 2 : 1)
            for j in z:(c - 1)
                u = Q[:, j]
                axpy!(-dot(u, v), u, v)
            end
        end
        nv = norm(v)
        # The threshold is relative to the original column, not to `v`: by this point the
        # deferred flush has already projected `v` against every earlier block.
        (iszero(nv) || nv <= rtol * norm(A[:, c])) &&
            throw(ArgumentError("column $c is a combination of the columns before it"))
        v ./= nv
        copyto!(Q[:, c], v)
    end
    R = Matrix{T}(undef, n, n)
    matmul!(R, Q', A, one(T), zero(T))
    for j in 1:n, i in (j + 1):n
        R[i, j] = zero(T)
    end
    return UpdatableQR(Q, R; capacity)
end
```

- [ ] **Step 4: Run the tests to verify they pass**

Expected: PASS, all five items. The reorthogonalization item only means anything while the
fixture stays in the regime `cond(A) * eps << 1`, which is where one pass restores orthogonality
to order `eps`. Raising `n` leaves that regime and breaks the item rather than strengthening it:
the same Vandermonde fixture measures `cond(A)` of 8.8e8 at `n = 12`, 5.2e10 at `n = 14` and
1.3e19 at `n = 40`, and by `n = 40` neither setting is orthogonal at all. At `n = 12`, `s = 8`
the two values are 2.3e-4 and 2.3e-12; at `n = 14`, `s = 8` they are 5.5e-3 and 8.0e-10. If the
item ever needs retuning, move `n` between 12 and 14 and record the measured `cond` and both
values in the comment.

- [ ] **Step 5: Record the deviation in the decision ledger**

Append to `.superpowers/sdd/2026-09-08-m1-camarero-construction/progress.md`, in the format the
M0/M2a ledger uses (ruling, reason, cost if wrong):

- `lu_crout` defaults to `pivot = RowMaximum()` where spec section 4 writes `NoPivot()`, so that
  the default matches `LinearAlgebra.lu` and a caller who does not think about pivoting gets the
  stable factorization. Cost if wrong: callers who want the unpivoted form must say so.

- [ ] **Step 6: Commit**

```bash
git add src/construct.jl test/construct.jl .superpowers/sdd/2026-09-08-m1-camarero-construction/progress.md
git commit -m "Add block classical Gram-Schmidt QR construction"
```

---

### Task 6: The substituted flush is the one that runs

**Files:**
- Modify: `test/construct.jl`

**Interfaces:**
- Consumes: `cholesky_crout(A; s, rankk!)`, `lu_crout(A; s, matmul!)`, `qr_bcgs(A; s, matmul!)`
- Produces: nothing new; this task proves R6's seam is real and that the blocked path is reached at all

Without this, every residual test above passes for an implementation whose flush never fires,
because the unblocked left-looking loop is also correct.

- [ ] **Step 1: Write the failing test**

Append to `test/construct.jl`:

```julia
@testitem "the flush keyword arguments are the functions that run" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 12
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))

    calls = Ref(0)
    counting_rankk!(C, X, alpha, beta; uplo = 'L') =
        (calls[] += 1; UpdatableFactorizations.default_rankk!(C, X, alpha, beta; uplo))
    F = cholesky_crout(A; s = 4, rankk! = counting_rankk!)
    @test calls[] == 2                       # flushes at columns 5 and 9
    @test norm(Matrix(F) - A) / norm(A) < 1.0e-13

    calls[] = 0
    counting_matmul!(C, X, Y, alpha, beta) = (calls[] += 1; mul!(C, X, Y, alpha, beta))
    A2 = randn(n, n)
    G = lu_crout(A2; s = 4, matmul! = counting_matmul!)
    # Two flushes, nine deferred column updates and eight deferred row updates. `Matrix(G)` is
    # rebuilt from `G.L` and `G.U`, so comparing against it would compare the factorization
    # with itself.
    @test calls[] == 19
    @test norm(G.L * G.U - A2[G.p, :]) / norm(A2) < 1.0e-12
end

@testitem "a full-block flush gives the same Cholesky factor as a rank-k flush" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 12
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))
    gemm_rankk!(C, X, alpha, beta; uplo = 'L') = mul!(C, X, X', alpha, beta)
    F = cholesky_crout(A; s = 4)
    G = cholesky_crout(A; s = 4, rankk! = gemm_rankk!)
    @test norm(Matrix(F.L) - Matrix(G.L)) / norm(Matrix(G.L)) < 1.0e-13
end
```

- [ ] **Step 2: Run the tests to verify they pass**

Expected: PASS. If either call count is off, the flush condition is not `c == z + s` with `z`
advanced to `c`; recount by hand for `n = 12`, `s = 4` before changing anything. For LU the
count is 2 flushes plus one call per column with `c > z` for the column update (9 of them) plus
one per column with `c > z` and `c < n` for the row update (8), which is 19.

- [ ] **Step 3: Commit**

```bash
git add test/construct.jl
git commit -m "Cover the substitutable flush seam"
```

---

### Task 7: Non-BLAS element types and derivative propagation

**Files:**
- Modify: `test/generic.jl`

**Interfaces:**
- Consumes: `cholesky_crout`, `lu_crout`, `qr_bcgs`
- Produces: nothing new; extends the existing `Float32`/`BigFloat`/`Dual` loop and the seeded-partial block

The existing item at `test/generic.jl:1-115` builds every non-BLAS matrix by broadcasting the
element type over a `Float64` matrix that is already positive definite, and separately re-runs
each operation with exactly one input carrying a nonzero `ForwardDiff.Partials` so the check is
not hollow. Both halves gain the three construction routines.

- [ ] **Step 1: Write the failing test**

Append to `test/generic.jl`:

```julia
@testitem "construction routines work for non-BLAS element types" begin
    using LinearAlgebra, ForwardDiff, Random
    Random.seed!(20260908)
    n = 7
    B64 = randn(n, n)
    A64 = Matrix(Symmetric(B64 * B64' + n * I))
    G64 = randn(n, n) + n * I
    for T in (Float32, BigFloat, ForwardDiff.Dual{Nothing, Float64, 1})
        # Float32 rounding does not scale with eps the way the others do, so this bound is
        # measured rather than derived: it sits above the worst residual this fixture produces
        # and below the smallest a broken flush produces.
        tol = T === Float32 ? 1.0f-4 : 100 * sqrt(eps(real(T)))
        A = T.(A64)
        F = cholesky_crout(A; s = 3)
        @test norm(Matrix(F) - A) / norm(A) < tol

        G = T.(G64)
        H = lu_crout(G; s = 3, pivot = NoPivot())
        @test norm(H.L * H.U - G) / norm(G) < tol

        Fq = qr_bcgs(T.(B64); s = 3)
        @test norm(Matrix(Fq) - T.(B64)) / norm(T.(B64)) < tol
        @test norm(Fq.Q' * Fq.Q - I) < tol
    end
end

@testitem "construction routines propagate derivatives" begin
    using LinearAlgebra, ForwardDiff, Random
    Random.seed!(20260908)
    n = 6
    T = ForwardDiff.Dual{Nothing, Float64, 1}
    seed(x, dir) = T.(x, ForwardDiff.Partials.(tuple.(dir)))
    partials(x) = ForwardDiff.partials.(x, 1)
    haspartial(x) = (ps = partials(x); any(!iszero, ps) && all(isfinite, ps))

    B64 = randn(n, n)
    A64 = Matrix(Symmetric(B64 * B64' + n * I))
    dir = Matrix(Symmetric(randn(n, n)))
    @test haspartial(Matrix(cholesky_crout(seed(A64, dir); s = 2).L))

    G64 = randn(n, n) + n * I
    @test haspartial(Matrix(lu_crout(seed(G64, randn(n, n)); s = 2, pivot = NoPivot()).U))

    Fq = qr_bcgs(seed(B64, randn(n, n)); s = 2)
    @test haspartial(Fq.Q)
    @test haspartial(Matrix(Fq.R))
end
```

- [ ] **Step 2: Run the tests to verify they pass or fail**

Expected: PASS. If the `BigFloat` Cholesky cell fails, the generic `default_rankk!` fallback is
not being reached — check that `Matrix{BigFloat}` is not a `StridedMatrix{<:BlasReal}`.

- [ ] **Step 3: Verify the derivative test is not hollow**

Temporarily change `seed(x, dir)` to `T.(x)` — discarding the seeded partial — and re-run the
second item. Every one of its four assertions must fail. Restore `seed` afterwards.

Then set the `Float32` bound the way `test/generic.jl`'s existing tolerance was set: from both
sides, not only from the passing run.

1. Print every `Float32` residual and orthogonality value the first item computes and take the
   worst.
2. Break the flush deliberately — drop the sign on the deferred update in `cholesky_crout`, so
   `rankk!` is called with `one(T)` instead of `-one(T)` — and record the smallest `Float32`
   value the item then produces.
3. Choose a bound that sits above the first and below the second, restore the flush, and
   replace the two descriptive phrases in the comment with the two measured numbers. If no such
   bound exists, the item is not discriminating and needs a stiffer fixture, not a looser
   bound.

- [ ] **Step 4: Commit**

```bash
git add test/generic.jl
git commit -m "Cover non-BLAS element types in construction"
```

---

### Task 8: Return-type concreteness gate

**Files:**
- Modify: `test/strict.jl`

**Interfaces:**
- Consumes: `Base.infer_return_type`, the three construction routines
- Produces: nothing new

`StrictMode` stays out of `Project.toml`: this milestone adds no in-source assertion, and the
existing gate is `StrictModeTest` in the test environment plus explicit `@allocated` checks.
`@test_noalloc` is not applicable here — these routines allocate their factors by definition —
so the gate is the inferred return type. Spec requirement R10 targets the `DenseQ` rank-1
kernels and belongs to M2b; nothing in this milestone changes that.

- [ ] **Step 1: Write the failing test**

Append to `test/strict.jl`:

```julia
@testitem "construction routines have concrete return types" begin
    using LinearAlgebra
    chol(A) = cholesky_crout(A; s = 8)
    lup(A) = lu_crout(A; s = 8)
    lun(A) = lu_crout(A; s = 8, pivot = NoPivot())
    qrb(A) = qr_bcgs(A; s = 8)
    for M in (Matrix{Float64}, Matrix{ComplexF64})
        @test isconcretetype(Base.infer_return_type(chol, Tuple{M}))
        @test isconcretetype(Base.infer_return_type(lup, Tuple{M}))
        @test isconcretetype(Base.infer_return_type(lun, Tuple{M}))
        @test isconcretetype(Base.infer_return_type(qrb, Tuple{M}))
    end
end
```

- [ ] **Step 2: Run the tests to verify they pass or fail**

Expected: PASS. A failure means a branch returns two different types — the most likely cause is
a `pivot` value being read at run time instead of dispatched on, so re-check that `_pivotrow`
selects by type.

- [ ] **Step 3: Commit**

```bash
git add test/strict.jl
git commit -m "Gate construction return types on concreteness"
```

---

### Task 9: Benchmark harness

**Files:**
- Create: `bench/Project.toml`, `bench/construct_sweep.jl`, `bench/README.md`, `bench/results/.gitkeep`

**Interfaces:**
- Consumes: `cholesky_crout`, `lu_crout`, `qr_bcgs`, `default_rankk!`, `LinearAlgebra.cholesky!`, `LinearAlgebra.lu!`, `LinearAlgebra.qr!`
- Produces: `bench/construct_sweep.jl` writing `bench/results/construction-<hostname>.json`

- [ ] **Step 1: Create the benchmark environment**

From an MCP Julia session, with the working directory set to the package root — both paths below
are relative to it:

```julia
using Pkg
Pkg.activate("bench")
Pkg.develop(path = ".")
Pkg.add(["Chairmarks", "JSON", "ForwardDiff", "LinearAlgebra", "Random", "Printf", "Dates"])
Pkg.compat("Chairmarks", "1.3")
Pkg.compat("JSON", "1.6")
Pkg.compat("ForwardDiff", "1.4")
Pkg.compat("LinearAlgebra", "1.12")
Pkg.compat("Random", "1.11")
Pkg.compat("Printf", "1.11")
Pkg.compat("Dates", "1.11")
Pkg.resolve()
```

Every module the script loads is declared here, including the four standard libraries. A
standard library that is not in `[deps]` resolves in the default environment and fails only
under `--project=bench`, which is how the sweep is run.

`QRupdatesFast` is not added here. It is a GPL-linked comparison for the updating verbs and
belongs to M4's benchmark task, not to this one.

- [ ] **Step 2: Write `bench/construct_sweep.jl`**

```julia
# Sweep of the construction layer against the standard library. Writes every sample, not just
# the median, so the plots and tables can be regenerated without re-running.
using UpdatableFactorizations
using LinearAlgebra, ForwardDiff, Chairmarks, JSON, Random, Printf, Dates

BLAS.set_num_threads(1)
Random.seed!(20260908)

const ROWS = Dict{String, Any}[]

function record(; routine, variant, eltype, n, s, bench, relerr::Float64,
        extra = Dict{String, Any}())
    times = [smp.time for smp in bench.samples]
    row = Dict{String, Any}(
        "routine" => routine, "variant" => variant, "eltype" => string(eltype),
        "n" => n, "s" => s, "median_seconds" => median(times), "samples" => times,
        "relerr" => relerr,
    )
    merge!(row, extra)
    push!(ROWS, row)
    @printf("%-10s %-16s %-8s n=%-5d s=%-5s %8.1f ms  relerr %.1e\n",
        routine, variant, string(eltype), n, s === nothing ? "-" : string(s),
        1.0e3 * median(times), relerr)
    return row
end

spd(T, n) = (B = randn(T, n, n); Matrix(Hermitian(B * B' + n * I)))
median(v) = sort(v)[(length(v) + 1) ÷ 2]

# `norm` over a Dual matrix returns a Dual, which has no Float64 conversion; the value is the
# part the sweep records.
scalar(x) = Float64(x)
scalar(x::ForwardDiff.Dual) = Float64(ForwardDiff.value(x))

function cholesky_cells(ns, ss)
    for n in ns
        A = spd(Float64, n)
        record(; routine = "cholesky", variant = "lapack potrf", eltype = Float64, n, s = nothing,
            bench = @be(copy(A), cholesky!(Symmetric(_, :L)), evals = 1),
            relerr = norm(Matrix(cholesky(Symmetric(A, :L))) - A) / norm(A))
        gemm_rankk!(C, X, alpha, beta; uplo = 'L') = mul!(C, X, X', alpha, beta)
        for s in ss, (name, f) in (("rankk", default_rankk!), ("gemm", gemm_rankk!))
            s < n || continue
            F = cholesky_crout(A; s, rankk! = f)
            record(; routine = "cholesky", variant = name, eltype = Float64, n, s,
                bench = @be(cholesky_crout(A; s, rankk! = f), evals = 1),
                relerr = norm(Matrix(F) - A) / norm(A))
        end
    end
    return
end

function lu_cells(ns, ss)
    dominant = Dict{String, Any}("matrix" => "dominant")
    random = Dict{String, Any}("matrix" => "random")
    for n in ns
        # Two fixtures. The diagonally dominant one has nonzero leading minors, which is what the
        # unpivoted factorization needs to be accurate, but partial pivoting picks the diagonal
        # at every column and so never swaps a row. The plain random one pivots, and it is the
        # only fixture on which the cost of pivoting is visible.
        Ad = Matrix(randn(n, n) + n * I)
        Ar = randn(n, n)
        record(; routine = "lu", variant = "lapack getrf", eltype = Float64, n, s = nothing,
            bench = @be(copy(Ar), lu!(_), evals = 1),
            relerr = norm(Matrix(lu(Ar)) - Ar) / norm(Ar), extra = random)
        record(; routine = "lu", variant = "stdlib nopivot", eltype = Float64, n, s = nothing,
            bench = @be(copy(Ad), lu!(_, NoPivot()), evals = 1),
            relerr = norm(Matrix(lu(Ad, NoPivot())) - Ad) / norm(Ad), extra = dominant)
        for s in ss
            s < n || continue
            F = lu_crout(Ad; s, pivot = NoPivot())
            record(; routine = "lu", variant = "crout nopivot", eltype = Float64, n, s,
                bench = @be(lu_crout(Ad; s, pivot = NoPivot()), evals = 1),
                relerr = norm(F.L * F.U - Ad) / norm(Ad), extra = dominant)
            # The pivoted and unpivoted timings on the same pivoting matrix. The unpivoted
            # residual here is not a quality number: without pivoting a random matrix has no
            # stability bound, and only the time is being compared.
            H = lu_crout(Ar; s, pivot = NoPivot())
            record(; routine = "lu", variant = "crout nopivot", eltype = Float64, n, s,
                bench = @be(lu_crout(Ar; s, pivot = NoPivot()), evals = 1),
                relerr = norm(H.L * H.U - Ar) / norm(Ar), extra = random)
            G = lu_crout(Ar; s)
            record(; routine = "lu", variant = "crout rowmaximum", eltype = Float64, n, s,
                bench = @be(lu_crout(Ar; s), evals = 1),
                relerr = norm(G.L * G.U - Ar[G.p, :]) / norm(Ar),
                extra = merge(random, Dict{String, Any}("interchanges" => count(G.p .!= 1:n))))
        end
    end
    return
end

function qr_cells(ns, ss)
    for n in ns
        A = randn(n, n)
        F = qr(A); Qh = Matrix(F.Q)
        record(; routine = "qr", variant = "lapack geqrf", eltype = Float64, n, s = nothing,
            bench = @be(copy(A), qr!(_), evals = 1), relerr = norm(Qh * F.R - A) / norm(A),
            extra = Dict{String, Any}("ortherr" => norm(Qh' * Qh - I)))
        record(; routine = "qr", variant = "lapack geqrf+Q", eltype = Float64, n, s = nothing,
            bench = @be(copy(A), (G = qr!(_); Matrix(G.Q)), evals = 1),
            relerr = norm(Qh * F.R - A) / norm(A),
            extra = Dict{String, Any}("ortherr" => norm(Qh' * Qh - I)))
        for s in ss, reorth in (false, true)
            s < n || continue
            Fq = qr_bcgs(A; s, reorth)
            record(; routine = "qr", variant = "bcgs reorth=$reorth", eltype = Float64, n, s,
                bench = @be(qr_bcgs(A; s, reorth), evals = 1),
                relerr = norm(Matrix(Fq) - A) / norm(A),
                extra = Dict{String, Any}("ortherr" => norm(Fq.Q' * Fq.Q - I)))
        end
    end
    return
end

# Does blocking buy anything when mul! is itself a scalar triple loop? Compared against the same
# routine with s >= n, which never flushes and so is the unblocked left-looking factorization.
function nonblas_cells()
    for (T, n) in ((BigFloat, 120), (ForwardDiff.Dual{Nothing, Float64, 1}, 200))
        A = T.(spd(Float64, n))
        for s in (16, 64, n)
            F = cholesky_crout(A; s)
            record(; routine = "cholesky", variant = s == n ? "unblocked" : "rankk",
                eltype = T, n, s, bench = @be(cholesky_crout(A; s), evals = 1, seconds = 30),
                relerr = scalar(norm(Matrix(F) - A) / norm(A)))
        end
        record(; routine = "cholesky", variant = "stdlib generic", eltype = T, n, s = nothing,
            bench = @be(cholesky(Hermitian(A, :L)), evals = 1, seconds = 30),
            relerr = scalar(norm(Matrix(cholesky(Hermitian(A, :L))) - A) / norm(A)))
    end
    return
end

cholesky_cells((2000, 4000), (64, 128, 256))
lu_cells((2000, 4000), (64, 128, 256))
qr_cells((2000, 4000), (32, 64, 128, 256))
nonblas_cells()

meta = Dict{String, Any}(
    "host" => gethostname(), "julia" => string(VERSION), "blas" => BLAS.get_config() |> string,
    "blas_threads" => BLAS.get_num_threads(), "date" => string(Dates.today()),
    "rows" => ROWS,
)
mkpath(joinpath(@__DIR__, "results"))
open(joinpath(@__DIR__, "results", "construction-$(gethostname()).json"), "w") do io
    JSON.print(io, meta)
end
```

Add `using Dates` to the header line when writing the file.

- [ ] **Step 3: Smoke-run it small**

Temporarily call `cholesky_cells((200,), (16,))`, `lu_cells((200,), (16,))`,
`qr_cells((200,), (16,))` and skip `nonblas_cells()`. Run with
`julia --project=bench bench/construct_sweep.jl`.
Expected: a table prints, every `relerr` is below 1e-13, and a JSON file appears under
`bench/results/`. Restore the full sizes afterwards.

- [ ] **Step 4: Write `bench/README.md`**

State: what the sweep measures, the exact command, where the results land, that plots are
regenerated from the saved JSON rather than by re-running, and that these algorithms are slower
than the LAPACK routines they are compared against.

- [ ] **Step 5: Commit**

```bash
git add bench/
git commit -m "Add the construction benchmark sweep"
```

---

### Task 10: The R14 gate

**Files:**
- Create: `bench/results/construction-<host>.json`
- Modify: `docs/superpowers/specs/RESULTS.md`

**Interfaces:**
- Consumes: `bench/construct_sweep.jl`
- Produces: committed datapoints; the two previously unmeasured claims settled

All benchmark gates run on this development machine. There is no remote benchmark host for this
project. Every ratio the sweep reports is measured by interleaving both arms within one round, so
clock drift cancels between them; absolute times and comparisons across separate runs are not
comparable and never were, regardless of which machine produced them.

- [ ] **Step 1: Instantiate and run**

```bash
julia --project=bench -e 'using Pkg; Pkg.instantiate()'
julia --project=bench bench/construct_sweep.jl
```

Leave cores free for other work; the script is single-threaded by construction.

- [ ] **Step 2: Check the gate**

The gate passes when all four hold. It is not a speed gate — these algorithms lose — it is a
reproducibility and accuracy gate:

1. Every `relerr` for Cholesky and LU is below 1e-15, and for QR below 1e-13. The one cell
   exempt is `crout nopivot` on the random matrix, which exists for its timing: an unpivoted
   factorization of a matrix with no diagonal dominance has no stability bound, so its residual
   is recorded and not gated.
2. Every Float64 ratio against its LAPACK baseline is within 15% of a same-shape re-run of
   `docs/superpowers/specs/alg1.jl` and `alg23.jl` on this machine. The comparison is against
   those scripts and not against `RESULTS.md`, which times the bare kernels rather than the
   shipped entry points. `bench/construct_sweep.jl` times the shipped entry points, which also
   copy `A` into a fresh working array, allocate the factor, and build the `UpdatableCholesky`
   or `UpdatableLU` wrapper around it — that construction overhead is part of what a caller
   pays, so it stays in the timing, and the ratios it produces are expected to sit below the
   spike's. A cell outside the band against the same-machine spike is investigated before the
   gate is called passed.
3. The unpivoted `lu_crout` versus `lu!(A, NoPivot())` ratio, both on the diagonally dominant
   matrix, is above 10x at both sizes.
4. The two previously unmeasured cells produce a number, whatever it is.

- [ ] **Step 3: Write the two new results into `RESULTS.md`**

Append a section headed `# M1 gate, <host>` containing:

- the host, Julia version, BLAS configuration, thread count and date;
- the pivoted `lu_crout` ratio against `getrf` at both sizes, next to the unpivoted ratio on the
  same random matrix and the number of row interchanges that matrix produced, with a
  one-sentence statement of whether pivoting costs anything measurable. The pivoted and
  unpivoted times must come from the same fixture: on the diagonally dominant matrix partial
  pivoting selects the diagonal at every column, performs no interchange, and answers nothing;
- the `BigFloat` and `Dual` cells, with a one-sentence verdict on whether blocking helps a
  non-BLAS element type — the honest answer is whatever the `s = 16` and `s = 64` medians say
  relative to the `s = n` unblocked cell, and if it does not help, say so;
- a restatement that no cell beats blocked LAPACK.

Write only what the JSON says. Do not carry a number forward from the earlier spike run.

- [ ] **Step 4: Commit**

```bash
git add bench/results docs/superpowers/specs/RESULTS.md
git commit -m "Record the construction gate"
```

---

### Task 11: Documentation

**Files:**
- Create: `docs/src/construction.md`
- Modify: `docs/make.jl`, `docs/src/provenance.md`, `docs/src/index.md`

**Interfaces:**
- Consumes: the docstrings written in Tasks 1-5, the JSON from Task 10
- Produces: a construction page in the built site

- [ ] **Step 1: Write `docs/src/construction.md`**

```markdown
# Construction

Three blocked left-looking factorizations from Camarero, arXiv:1812.02056. Each forms one column
at a time, defers the trailing update, and flushes it every `s` columns through a function you
can replace.

```julia
using UpdatableFactorizations, LinearAlgebra

A = let B = randn(400, 400); Matrix(Symmetric(B * B' + 400I)) end
F = cholesky_crout(A; s = 64)      # an UpdatableCholesky, ready to update
G = lu_crout(randn(400, 400); s = 64)   # partial pivoting, as LinearAlgebra.lu does
H = qr_bcgs(randn(600, 400); s = 64)    # an UpdatableQR, ready to update
```

## These are slower than LAPACK

The standard library calls LAPACK's blocked `potrf`, `getrf` and `geqrf`. Measured
single-threaded on one core, as a ratio against those routines where anything above 1.00 would
be a win:

| routine | flush | baseline | n=2000 | n=4000 |
| --- | --- | --- | --- | --- |
| `cholesky_crout` | symmetric rank-k, the default | `potrf` | 0.88 | 0.93 |
| `cholesky_crout` | full-block multiply | `potrf` | 0.58 | 0.63 |
| `lu_crout`, unpivoted | full-block multiply | `getrf`, pivoted | 0.95 | 0.91 |
| `qr_bcgs` | full-block multiply | `geqrf` plus forming thin Q | 0.92 | 0.93 |

Use `LinearAlgebra.cholesky`, `lu` and `qr` when you want the fastest factorization. These
routines exist because the flush is substitutable, because they extend to element types LAPACK
does not have, and for the one case below.

There is one exception. `lu!(A, NoPivot())` in the standard library is an unblocked generic
fallback running at 0.05 to 0.06 of pivoted `getrf`, so `lu_crout(A; pivot = NoPivot())` is
14.7x faster at n=2000 and 17.8x faster at n=4000. That is a gap in the standard library, not a
win over LAPACK.

Routing the deferred update through a Strassen-Winograd multiply makes Cholesky slower, not
faster: 0.63 down to 0.47 at n=4000. The flush is a rank-`s` product whose inner dimension is
`s`, so the flop cut does not pay for the extra buffer traffic.

## Accuracy

`cholesky_crout` reaches `norm(L*L' - A)/norm(A)` of 1.3e-16 to 2.5e-16, which is LAPACK's
range. `lu_crout` reaches `norm(L*U - A[p, :])/norm(A)` of 1.6e-16 to 2.8e-16, against 9.0e-17
to 1.0e-16 for pivoted `getrf` — about twice LAPACK's residual, and a long way from a problem.

`qr_bcgs` does not. Block classical Gram-Schmidt loses orthogonality as the square of the
condition number:

| quantity | `qr` (Householder) | `qr_bcgs` |
| --- | --- | --- |
| `norm(Q*R - A)/norm(A)` | 1.2e-15 | 2.3e-14 to 9.7e-14 |
| `norm(Q'Q - I)` | 6.8e-14 to 1.0e-13 | 1.2e-12 to 7.2e-12 |

`reorth = true`, the default, runs the projection a second time. That gives orthogonality of
order `eps` provided the condition number times `eps` is well below one, at roughly twice the
cost. `qr_householder`, which calls `LinearAlgebra.qr` and wraps the result, is unconditionally
more accurate than either setting and is the recommended path.

## Substituting the flush

`cholesky_crout` takes `rankk!(C, A, alpha, beta; uplo)` and computes `alpha*A*A' + beta*C`.
`lu_crout` and `qr_bcgs` take `matmul!(C, A, B, alpha, beta)` with the meaning of
`LinearAlgebra.mul!`. Both are plain function-valued keyword arguments: there is no abstract
type, no trait and no package extension.

```julia
counted = Ref(0)
mine!(C, A, B, alpha, beta) = (counted[] += 1; mul!(C, A, B, alpha, beta))
lu_crout(randn(200, 200); s = 64, matmul! = mine!)
```

The Cholesky default writes only one triangle, through `syrk!` for real element types and
`herk!` for complex ones. Passing a full-block multiply instead is the `gemm` row of the table
above, and costs between 1.2x and 2.1x.

## Failure

`cholesky_crout` throws `PosDefException(c)` at the first column with a non-positive pivot.
`lu_crout` throws `ZeroPivotException(c)`; with `pivot = NoPivot()` that means the leading
principal minor is singular, and `rtol` widens the test from exact zero to a threshold relative
to the column's norm. `qr_bcgs` throws when a column's projection collapses onto the columns
before it, which means the matrix is rank deficient; it takes the same `rtol`, and needs it,
because a rank-deficient input in floating point leaves rounding noise rather than an exact
zero.
```

- [ ] **Step 2: Add the page to `docs/make.jl`**

Insert `"Construction" => "construction.md",` between `"Getting started"` and
`"Updating and downdating"`.

- [ ] **Step 3: Check the API page picks up the new names**

`docs/src/api.md` is an `@autodocs` block over the whole module and is not edited. After the
build in Step 7, confirm `cholesky_crout`, `lu_crout`, `qr_bcgs` and `default_rankk!` all appear
on the API page. A `public` but unexported name is included by `@autodocs`; if `default_rankk!`
is missing, the `public` declaration in `src/UpdatableFactorizations.jl` is what to check.

- [ ] **Step 4: Extend `docs/src/provenance.md`**

Add three rows to the routines table:

| routine | article |
| --- | --- |
| Blocked Crout Cholesky | Camarero, *Simple, Fast and Practicable Algorithms for Cholesky, LU and QR Decomposition Using Fast Rectangular Matrix Multiplication*, arXiv:1812.02056 (2018), Algorithm 1 |
| Blocked Crout LU | Camarero, arXiv:1812.02056 (2018), Algorithm 2; the partial-pivoting protocol follows Golub and Van Loan, *Matrix Computations*, 4th edition, section 3.4 |
| Block classical Gram-Schmidt QR | Camarero, arXiv:1812.02056 (2018), Algorithm 3; the reorthogonalization bound is Giraud, Langou and Rozložník, *The loss of orthogonality in the Gram-Schmidt orthogonalization process*, Computers and Mathematics with Applications 50 (2005), 1069-1075 |

The titles here and in the three docstrings must be the same string.

- [ ] **Step 5: Mention construction on the home page**

Add one paragraph to `docs/src/index.md` saying the package also builds factorizations with the
blocked Crout and block-Gram-Schmidt algorithms, that those are slower than the standard
library's LAPACK calls, and linking the construction page. Make no speed claim.

- [ ] **Step 6: Run every code block on the construction page**

Paste each `julia` block from `docs/src/construction.md` into an MCP Julia session against the
real package and run it. Every one must run to completion with no error. Documenter does not
execute these blocks, so nothing else catches an example that throws.

- [ ] **Step 7: Build the docs**

```bash
julia --project=docs docs/make.jl
```

Expected: builds with no missing-docstring and no broken-link warning, and the four new names
from Step 3 appear on the API page.

- [ ] **Step 8: Commit**

```bash
git add docs/
git commit -m "Document the construction layer"
```

---

### Task 12: Full-suite gate

**Files:**
- Modify: none expected

- [ ] **Step 1: Run the whole suite warm**

`julia_run_testitems` over the package with `max_workers = 4`, no filter.
Expected: every item passes, including the 32 items shipped before this milestone.

- [ ] **Step 2: Run `Pkg.test()` once, cold**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

This is the pre-commit gate and the only place a cold Julia process is warranted. The warm
runner masks a missing test dependency; this catches it.

- [ ] **Step 3: Format**

Run `runic` over `src/`, `test/` and `bench/`. Commit any reformatting separately.

- [ ] **Step 4: Check the diff's comments and prose**

Re-read every comment, docstring and documentation paragraph added by this milestone. Two
defects to hunt:

- any reference to this plan, a milestone, a task number, a review, or how the code came to be;
- any sentence that implies these routines are fast, competitive with, or better than LAPACK,
  outside the single unpivoted-LU statement, which is a statement about the standard library's
  missing blocked path.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Format and finish the construction layer"
```

---

## Self-review

**Spec coverage for this milestone.** R2 (Tasks 2, 3, 5), R3 (Tasks 3 and 4), R4 (Task 5), R6
(Tasks 1 and 6), R14 (Tasks 9 and 10), and the construction half of R11 (Task 7), R12 and R15
(Task 11). R1, R7, R9 and R10 were closed or scoped by M0/M2a. R8 belongs to M3, R13 to M4.

**R5 belongs to M2b.** `qr_householder` is a delegation to `LinearAlgebra.qr` wrapped in an
`UpdatableQR`, so it ships with the type it returns rather than with the three algorithms this
milestone implements. Nothing here defines it; the construction page links to it as the
accuracy-preferring alternative to `qr_bcgs`.

**`lu_crout` defaults to `pivot = RowMaximum()`.** Spec section 4 writes `NoPivot()`. The
default here matches `LinearAlgebra.lu`, so a caller who does not think about pivoting gets the
stable factorization rather than one that throws on a zero leading minor. The unpivoted form is
one keyword away and is the form the one measured win belongs to, so nothing is hidden by this.
Task 5 Step 5 writes this into the decision ledger too.

**`qr_bcgs` takes `rtol`, which the spec does not list.** The rank-deficiency guard compares the
projected column norm against the column's own norm. An exact-zero test fires only for exactly
dependent columns; a rank-deficient input in floating point leaves noise of order `eps` times
the column norm, and without `rtol` the routine would return a `Q` whose column is amplified
noise with no signal to the caller. `lu_crout` carries the same keyword for the same reason. The
default is 0, so the behavior without it is the exact-zero test.

**`cholesky_crout` requires a `Symmetric` or `Hermitian` argument to store the triangle `uplo`
names.** `Matrix(Hermitian(A, uplo))` throws when they disagree, so `cholesky_crout(Symmetric(B
* B'))`, whose wrapper stores the upper triangle, needs `uplo = :U`. This is stated in the
docstring and the documented examples pass plain matrices. Making the default follow the
wrapper's own `uplo` is a possible M2b convenience; it is not done here because the current
behavior fails loudly and the spec fixes the default at `:L`.

**`StrictMode` stays out of `[deps]`.** Spec section 8 lists it; commit 4d6c4ad removed it as
unused, and this milestone adds no in-source assertion that would need it. The gate for the
construction routines is inferred-return-type concreteness (Task 8), not `@assert_noalloc`,
because these routines allocate their factors by definition. R10's `@assert_noalloc` and
`@assert_typestable` target the `DenseQ` rank-1 kernels and remain M2b's to decide.

**What the flush seam does not cover.** `lu_crout` and `qr_bcgs` share one `matmul!` keyword
each, but `qr_bcgs` calls it four times per flush with two different shapes (`BQ' * BM` and
`BQ * C`) and once at the end for `Q' * A`. A substituted `matmul!` therefore sees more than the
deferred update alone. This is stated in the docstring rather than split into more keywords.

**Accuracy note for the docs.** The measured `qr_bcgs` orthogonality of 1.2e-12 to 7.2e-12 is
from `reorth = false` on well-conditioned random matrices at n=2000 and n=4000. The
`reorth = true` figure has not been measured at those sizes — Task 10's sweep records both, and
until it runs, the documentation states the bound (`O(eps)` when the condition number times
`eps` is well below one) as a citation to Giraud, Langou and Rozložník, not as a measurement of
this implementation.

**One accuracy claim that must be re-checked, not assumed.** `lu_crout` reconstructs the
standard library's packed `factors` array by multiplying each row of the unit-upper factor by
its pivot, and `UpdatableLU` immediately divides it back out. That round trip is exact in
neither direction, and costs about one unit in the last place. The 1e-12 relative tolerances in
Tasks 3 and 4 have several decades of headroom over that, but a future implementer who tightens
those tolerances to 1e-15 will hit it.
