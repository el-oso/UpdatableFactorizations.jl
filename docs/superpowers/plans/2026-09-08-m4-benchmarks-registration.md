# M4: benchmarks, documentation and registration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish measured, reproducible benchmarks of every updating verb against the standard library, the three MIT prior-art packages and the GPL `qrupdate-ng` wrapper, finish the eight-page documentation site, and register the package in General.

**Architecture:** `bench/` is a separate environment that the package never depends on. One shared `bench/harness.jl` holds the sample recorder, the JSON writer, the single-thread/LP64 BLAS setup and the clock-lock guard; two sweep scripts (`construct_sweep.jl` from M1, `updating_sweep.jl` from this milestone) include it and write every sample to `bench/results/*.json`. A third script reads only those JSON files and regenerates the plots and the benchmarks page, so the published numbers can never drift from the recorded ones and no reader has to re-run anything.

**Tech Stack:** Julia 1.12, `Chairmarks` 1.3, `JSON` 1.8, `CairoMakie` 0.15, `OpenBLAS32_jll` 0.3.33, `QRupdatesFast` 1.0, `QRupdate` 1.0, `UpdatableQRFactorizations` 1.0, `UpdatableCholeskyFactorizations` 1.1 — all confined to `bench/Project.toml`. DocumenterVitepress for the site.

**Spec:** `docs/superpowers/specs/2026-09-08-updatablefactorizations-design.md`, sections 8, 9 and 12.

**Precondition:** M0, M2a, M2b, M1 and M3 are complete and committed. This plan benchmarks the verbs those milestones ship and depends on their public signatures:
`lowrankupdate!(F::UpdatableCholesky, v)`, `lowrankdowndate!(F::UpdatableCholesky, v)`,
`delete_column!(F, j)`, `insert_column!(F, j, x)`, `shift_columns!(F, i, j)`,
`lowrankupdate!(F::UpdatableLU, u, v; rtol)`,
`UpdatableQR(A; capacity)` with `F.Q`, `F.R`, `lowrankupdate!(F, u, v)`,
`insert_column!(F, j, x; rtol)`, `delete_column!(F, j)`, `shift_columns!(F, i, j)`,
`insert_row!(F, i, x)`, `delete_row!(F, i; rtol)`, `Matrix(::UpdatableQR)`,
and the constructor form that selects a `Q` representation — spelled `UpdatableQR(qr(A), GivensQ)`
throughout this plan. That spelling is confirmed against the shipped package in Task 4, Step 0
before any QR cell is written, because the milestone that ships it may spell it otherwise.
M1 has already created `bench/Project.toml`, `bench/construct_sweep.jl`, `bench/README.md` and `bench/results/`.

**Verified against the real comparison packages.** Everything below was run before this plan was written, on `neuromancer`, Julia 1.12.7, against the registered versions named above. The code in the tasks is that code.

- `QRupdatesFast` 1.0.1 wraps **three** `qrupdate-ng` routines, not thirteen: `qrshc!` (QR column shift, via the public `qrshift`/`qrshift!`), `lu1up!` (unpivoted LU rank-1) and `lup1up!` (pivoted LU rank-1). Nothing else in the catalogue has a `QRupdatesFast` comparison.
- `libqrupdate` calls the **LP64** BLAS interface (`dtrsv_`, `daxpy_`, `drot_`, `dlartg_`). Julia forwards only ILP64 `libopenblas64_` into libblastrampoline, so `lup1up!` and `qrshc!` print `Error: no BLAS/LAPACK library loaded for dtrsv_()` and return silently wrong answers — measured residual 1.1 instead of 3e-16. Forwarding `OpenBLAS32_jll.libopenblas_path` with `clear = false` fixes it and leaves Julia's own ILP64 library in place. `lu1up!` calls no BLAS and is correct either way, which is exactly why this is easy to miss.
- `lup1up!`'s permutation argument is `Vector{Int32}` and is **in/out**: it comes back permuted, and the identity that holds afterwards is `L*R == (A + u*v')[p, :]` with the *returned* `p`, in Julia's `lu` convention.
- `UpdatableQRFactorizations` 1.0.0 stores a **full m x m** `Q` — for a 40 x 10 problem `size(F.Q) == (40, 40)` — so it does strictly more work than a thin-Q update. Its constructor is `UpdatableGivensQR(A, r)` where `r` is the reserved column count; its verbs are `add_column!(F, x)` and `remove_column!(F, k)`.
- `QRupdate` 1.0.1 maintains **R only**, from the normal equations: `qraddcol(A, R, a)`, `qrdelcol(R, k)`, `qraddrow(R, a)` where `a` is a `1 x n` matrix. It never forms or updates `Q`, so it does strictly less work than a QR update.
- `UpdatableCholeskyFactorizations` 1.1.1 is `updatable_cholesky(A, m)` where `m` is the **total** capacity, not the headroom — passing `m < size(A, 1)` throws `DimensionMismatch`. Its verbs are `add_column!(C, a)` with `a` of length `n+1` and `remove_column!(C, i)`.
- Both `UpdatableQRFactorizations` and `UpdatableCholeskyFactorizations` export the names `UpdatableQR` and `UpdatableCholesky`. Loading them alongside this package makes both names ambiguous (`UndefVarError ... two or more modules export different bindings with this name`). Every benchmark call site must be module-qualified.
- Chairmarks' positional pipeline is `@be [[init] setup] f [teardown]`, resolved **by argument count**: with three positional arguments the third is `teardown`, not `f`. Only `f` is timed; `setup` runs once per sample outside the timing region. A mutating cell is therefore written `@be <setup> <f> evals = 1 samples = 100`, and `evals = 1` is not optional — without it Chairmarks reapplies the verb to an already-shrunken factorization and either times the wrong thing or throws `BoundsError`. A cell whose expression allocates its own result and mutates nothing — every `recompute` baseline, and `QRupdate`'s three routines — takes **one** positional argument, `@be <expr> evals = 1 samples = 100`, so the expression itself is what is timed. Writing such a cell as `@be <expr> identity ...` resolves as `(setup, f)`: the expression runs outside the timing region and the recorded median is the cost of `identity`.
- `neuromancer`'s governor reads `powersave`; `galen` (24 threads) and `wintermute` (12 threads) both read `performance`. The gate runs on one of the latter two.
- JSON 1.8 round-trips the row shape used here: `JSON.print(io, meta)` writes and `JSON.parsefile(path)` reads back a `JSON.Object{String, Any}` indexable by string.

## Global Constraints

- License MIT. No GPL code in the dependency graph outside `bench/`. `QRupdatesFast` is GPL-linked through `QRupdate_ng_jll` and may appear only in `bench/Project.toml` — never in `Project.toml` or `test/Project.toml`.
- Never read `qrupdate-ng` source. It is a benchmark target and a routine-name reference only.
- `julia = "1.12"`. No `@inbounds` anywhere. American spellings.
- Comments, docstrings, page text and commit messages state what is true of the code now. No reference to this plan, to a milestone, to a task number, or to how anything came to be.
- **No document claims a speed win over LAPACK.** Building a factorization from scratch goes through `LinearAlgebra`, and the construction layer is slower than the blocked LAPACK routine at every size measured. The pages state that in words and quote the ratios from the recorded construction sweep. The updating-versus-recomputing ratio is a different comparison and is reported as what it is: whether updating an existing factorization beats throwing it away and calling `cholesky`, `lu` or `qr` again.
- Every benchmark number on `docs/src/benchmarks.md`, including the construction ratios its preamble quotes, is generated by `bench/plot_results.jl` from a committed `bench/results/*.json` file. No number is typed by hand into that page or into the generator. `docs/src/construction.md` and `docs/src/q_representations.md` carry numbers from `construction-<host>.json` and `givensq_compaction-<host>.json`; those pages are hand-written, and whenever either sweep is re-run the numbers on them are re-derived from the new file.
- The benchmark gate runs on a clock-locked host. `neuromancer` is `powersave` and is never gate-authoritative.
- `Project.toml`, `test/Project.toml`, `bench/Project.toml` and `docs/Project.toml` are edited only through `Pkg` (`Pkg.add`, `Pkg.compat`) from an MCP Julia session. Never hand-write a UUID.
- Package tests run through the JuliaMCP `julia_run_testitems` tool with an explicit `max_workers`. A cold `Pkg.test()` is the pre-commit gate.
- Benchmark scripts are run cold, from the shell, one process per sweep. A warm session carries BLAS thread settings and compiled state between runs and has produced contradictory sweeps before.
- Do not post anything to GitHub — including the registration comment — without the user approving the exact text.

## File Structure

| file | responsibility |
| --- | --- |
| `bench/Project.toml` | gains `CairoMakie`, `OpenBLAS32_jll`, `QRupdate`, `QRupdatesFast`, `UpdatableCholeskyFactorizations`, `UpdatableQRFactorizations` |
| `bench/setup.jl` | develops the package into `bench/` and instantiates, so a fresh checkout runs |
| `bench/harness.jl` | `record!`, `save_results`, `median`, `single_threaded!`, `require_locked_clock`, `governor` |
| `bench/construct_sweep.jl` | the construction sweep, switched over to `harness.jl` |
| `bench/givensq_compaction.jl` | the compaction sweep, guarded by `harness.jl`, keeping its own writer |
| `bench/updating_sweep.jl` | the Cholesky, LU and QR updating cells |
| `bench/test_bench.jl` | the checks for `harness.jl` and for the sweep's own correctness assertions |
| `bench/plot_results.jl` | reads `bench/results/*.json`, writes the SVGs and `docs/src/benchmarks.md` |
| `bench/results/updating-<host>.json`, `bench/results/construction-<host>.json`, `bench/results/givensq_compaction-<host>.json` | every sample from the gate run, committed, one file per sweep |
| `bench/README.md` | how to reproduce from a fresh checkout |
| `docs/src/benchmarks.md` | generated page |
| `docs/src/assets/bench-*.svg` | generated plots |
| `docs/make.jl` | the eight-page `pages` list |
| `docs/src/index.md`, `README.md`, `docs/src/provenance.md` | brought up to what the package now ships |
| `.github/workflows/TagBot.yml` | tags releases once the package is registered |

## Benchmark case matrix

Published before the code. Every row is one `record!` call per size; `variant` is the string in the JSON.

| family | routine | `UpdatableFactorizations` | prior art | recompute baseline |
| --- | --- | --- | --- | --- |
| cholesky | `lowrankupdate!` | `lowrankupdate!(F, v)` | `LinearAlgebra.lowrankupdate!(C, copy(v))` | `cholesky(Symmetric(A + v*v', :L))` |
| cholesky | `lowrankdowndate!` | `lowrankdowndate!(F, v)` | `LinearAlgebra.lowrankdowndate!(C, copy(v))` | `cholesky(Symmetric(A - v*v', :L))` |
| cholesky | `delete_column!` | `delete_column!(F, 2)` | `UpdatableCholeskyFactorizations.remove_column!(C, 2)` | `cholesky(Symmetric(A[keep, keep], :L))` |
| cholesky | `append_column!` | `insert_column!(F, n+1, x)` | `UpdatableCholeskyFactorizations.add_column!(C, x)` | `cholesky(Symmetric(Ag, :L))` |
| cholesky | `insert_column!` | `insert_column!(F, 2, x)` | none | `cholesky(Symmetric(Ai, :L))` |
| cholesky | `shift_columns!` | `shift_columns!(F, 1, n)` | none | `cholesky(Symmetric(A[p, p], :L))` |
| lu | `lowrankupdate! nopivot` | `lowrankupdate!(F, u, v)` | `QRupdatesFast.lu1up!` | `lu(A + u*v', NoPivot())` |
| lu | `lowrankupdate! pivoted` | `lowrankupdate!(F, u, v)` | `QRupdatesFast.lup1up!` | `lu(A + u*v')` |
| qr | `lowrankupdate!` | `lowrankupdate!(F, u, v)` | none | `qr(A + u*v')` |
| qr | `insert_column!` | `insert_column!(F, n+1, x)` | `UpdatableQRFactorizations.add_column!`, `QRupdate.qraddcol` | `qr([A x])` |
| qr | `delete_column!` | `delete_column!(F, 2)` | `UpdatableQRFactorizations.remove_column!`, `QRupdate.qrdelcol` | `qr(A[:, keep])` |
| qr | `shift_columns!` | `shift_columns!(F, 1, n)` | `QRupdatesFast.qrshift!` | `qr(A[:, p])` |
| qr | `insert_row!` | `insert_row!(F, m+1, x)` | `QRupdate.qraddrow` | `qr([A; transpose(x)])` |
| qr | `delete_row!` | `delete_row!(F, 2)` | none | `qr(A[keep, :])` |

The four QR column verbs are additionally recorded against this package's implicit-rotation Q, as
`variant = "UpdatableFactorizations (GivensQ)"`. `insert_row!` and `delete_row!` throw on that
representation and have no such row.

Three comparisons are not like-for-like and carry a flag in the JSON so the page can say so:

- `QRupdate`'s three routines maintain `R` alone, from the normal equations. They never touch `Q`. Rows carry `"r_only" => true`.
- `UpdatableQRFactorizations` maintains a full `m x m` `Q` where this package maintains a thin `m x n` one. Rows carry `"full_q" => true`.
- Every `GivensQ` cell starts from a freshly built factorization, so it measures one verb against an empty rotation log. Apply cost rises with the log length, and that curve belongs to the compaction sweep. Rows carry `"fresh_log" => true`.

Sizes: Cholesky and LU at `n in (64, 128, 256, 512, 1024)`; QR at `(m, n) in ((256, 64), (512, 128), (1024, 256), (2048, 512))`. `Float64` throughout — the comparison libraries are `Float64`/`Float32` only, and element-type coverage is the test suite's job, not the benchmark's.

---

### Task 1: Benchmark environment, harness and the fresh-checkout path

**Files:**
- Create: `bench/harness.jl`, `bench/setup.jl`, `bench/test_bench.jl`
- Modify: `bench/Project.toml`, `bench/construct_sweep.jl`, `bench/README.md`

**Interfaces:**
- Consumes: `LinearAlgebra.BLAS`, `Chairmarks.@be`, `JSON.print`, `OpenBLAS32_jll.libopenblas_path`
- Produces: `record!(; family, routine, variant, eltype, n, m, s, bench, relerr, extra)`; `save_results(stem) -> String`; `median(v)`; `single_threaded!()`; `governor() -> String`; `require_locked_clock() -> String`; the global `ROWS::Vector{Dict{String,Any}}`

- [ ] **Step 1: Add the comparison packages to the benchmark environment**

From an MCP Julia session with `project` set to the package root:

```julia
using Pkg
Pkg.activate("bench")
Pkg.add([
    "CairoMakie", "OpenBLAS32_jll", "QRupdate", "QRupdatesFast",
    "UpdatableCholeskyFactorizations", "UpdatableQRFactorizations", "Test",
])
Pkg.compat("CairoMakie", "0.15")
Pkg.compat("OpenBLAS32_jll", "0.3.33")
Pkg.compat("Test", "1.11")
Pkg.compat("QRupdate", "1.0")
Pkg.compat("QRupdatesFast", "1.0")
Pkg.compat("UpdatableCholeskyFactorizations", "1.1")
Pkg.compat("UpdatableQRFactorizations", "1.0")
Pkg.resolve()
```

`CairoMakie`'s compat bound is the major-minor `Pkg.add` actually installed, not a version typed
from memory. Read it back before pinning it:

```julia
Pkg.status("CairoMakie")
```

`Test` is what `bench/test_bench.jl` loads and is the one standard library the environment does
not already declare; `LinearAlgebra`, `Random`, `Printf` and `Dates` are already in
`bench/Project.toml`. A standard library missing from `[deps]` resolves in the default
environment and fails only under `--project=bench`, which is how everything here is run. Confirm
all five are present:

```julia
using TOML
deps = keys(TOML.parsefile("bench/Project.toml")["deps"])
@assert issubset(["Dates", "LinearAlgebra", "Printf", "Random", "Test"], deps)
```

`QRupdatesFast` pulls `QRupdate_ng_jll`, which is GPL-3.0-or-later. Confirm it landed only here:

```julia
Pkg.activate("."); Pkg.status()      # no QRupdatesFast, no QRupdate_ng_jll
Pkg.activate("test"); Pkg.status()   # no QRupdatesFast, no QRupdate_ng_jll
```

- [ ] **Step 2: Write the failing check**

`bench/test_bench.jl`:

```julia
using Test, Random
include(joinpath(@__DIR__, "harness.jl"))

# Every cell function below builds its matrices with randn, and the checks assert a residual
# bound on the result. The seed makes that bound reproducible.
Random.seed!(20260908)

@testset "harness" begin
    @test median([3.0, 1.0, 2.0]) == 2.0
    @test median([4.0, 1.0, 3.0, 2.0]) == 2.0

    single_threaded!()
    @test BLAS.get_num_threads() == 1
    interfaces = [c.interface for c in BLAS.lbt_get_config().loaded_libs]
    @test :lp64 in interfaces      # libqrupdate resolves its BLAS symbols through this one
    @test :ilp64 in interfaces     # Julia's own linear algebra keeps the library it had

    @test governor() isa String

    empty!(ROWS)
    record!(;
        family = "probe", routine = "sum", variant = "Base", eltype = Float64, n = 8,
        bench = @be(zeros(8), sum, evals = 1, samples = 5), relerr = 0.0,
    )
    @test length(ROWS) == 1
    @test ROWS[1]["family"] == "probe"
    @test length(ROWS[1]["samples"]) == 5
    @test ROWS[1]["median_seconds"] > 0

    path = save_results("probe")
    meta = JSON.parsefile(path)
    @test meta["host"] == gethostname()
    @test length(meta["rows"]) == 1
    @test meta["rows"][1]["routine"] == "sum"
    rm(path)
    empty!(ROWS)
end
```

- [ ] **Step 3: Run it to verify it fails**

```bash
julia --project=bench bench/test_bench.jl
```

Expected: FAIL — `bench/harness.jl` does not exist. If it instead fails with
`ArgumentError: Package Test not found in current path`, Step 1 did not add `Test` to
`bench/Project.toml`.

- [ ] **Step 4: Write `bench/harness.jl`**

```julia
# Shared machinery for the benchmark sweeps: every sample is kept, not just a summary, so the
# plots and the published tables are regenerated from the saved file rather than by re-running.

using LinearAlgebra, Chairmarks, JSON, Dates, Printf
import OpenBLAS32_jll

const ROWS = Dict{String, Any}[]

median(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

# libqrupdate calls the LP64 BLAS interface. Julia forwards only its ILP64 OpenBLAS into
# libblastrampoline, which leaves those symbols unresolved: the calls print
# "no BLAS/LAPACK library loaded" and return a wrong factorization instead of failing. Forwarding
# an LP64 OpenBLAS alongside satisfies them; `clear = false` keeps Julia's ILP64 library in place.
function single_threaded!()
    BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path; clear = false)
    BLAS.set_num_threads(1)
    BLAS.get_num_threads() == 1 ||
        error("BLAS reports $(BLAS.get_num_threads()) threads; the sweep is single threaded")
    return nothing
end

function governor()
    path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    return isfile(path) ? strip(read(path, String)) : "unknown"
end

# A host whose clock is free to drift produces numbers that cannot be compared across runs, so a
# sweep on one is refused unless it is explicitly marked as exploratory.
function require_locked_clock()
    g = governor()
    g == "performance" && return g
    get(ENV, "ALLOW_UNLOCKED_CLOCK", "0") == "1" || error(
        "the CPU governor is \"$g\", not \"performance\"; run the sweep on a clock-locked " *
            "host, or set ALLOW_UNLOCKED_CLOCK=1 to record an exploratory run"
    )
    @warn "recording an exploratory run: the CPU governor is \"$g\""
    return g
end

function record!(;
        family, routine, variant, eltype, n, m = nothing, s = nothing, bench, relerr,
        extra = Dict{String, Any}()
    )
    times = [smp.time for smp in bench.samples]
    row = Dict{String, Any}(
        "family" => family, "routine" => routine, "variant" => variant,
        "eltype" => string(eltype), "n" => n, "m" => m, "s" => s,
        "median_seconds" => median(times), "samples" => times,
        "median_bytes" => median([smp.bytes for smp in bench.samples]),
        "relerr" => Float64(relerr),
    )
    merge!(row, extra)
    push!(ROWS, row)
    # `m` and `s` are family-specific — `m` is the row count of a QR problem, `s` the flush block
    # size of a construction sweep — and print as "-" for a family that has no such axis.
    @printf(
        "%-10s %-22s %-32s %-8s m=%-6s n=%-6d s=%-5s %9.3f ms  %8.0f B  relerr %.1e\n",
        family, routine, variant, string(eltype), m === nothing ? "-" : string(m), n,
        s === nothing ? "-" : string(s),
        1.0e3 * row["median_seconds"], row["median_bytes"], row["relerr"]
    )
    return row
end

function head_commit()
    repo = dirname(@__DIR__)
    return try
        readchomp(`git -C $repo rev-parse --short HEAD`)
    catch
        "unknown"
    end
end

function save_results(stem::AbstractString)
    meta = Dict{String, Any}(
        "host" => gethostname(), "governor" => governor(), "julia" => string(VERSION),
        "blas" => string(BLAS.get_config()), "blas_threads" => BLAS.get_num_threads(),
        "date" => string(Dates.today()), "commit" => head_commit(), "rows" => copy(ROWS),
    )
    dir = joinpath(@__DIR__, "results")
    mkpath(dir)
    path = joinpath(dir, "$stem-$(gethostname()).json")
    open(io -> JSON.print(io, meta), path, "w")
    println("wrote $path  ($(length(ROWS)) rows)")
    return path
end
```

- [ ] **Step 5: Run the check to verify it passes**

```bash
julia --project=bench bench/test_bench.jl
```

Expected: PASS, all assertions.

- [ ] **Step 6: Write `bench/setup.jl`**

A fresh checkout has no `bench/Manifest.toml`, and `Pkg.instantiate()` alone cannot resolve
`UpdatableFactorizations` from the working tree. This is the entry point the README names.

```julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = dirname(@__DIR__))
Pkg.instantiate()
Pkg.status()
```

- [ ] **Step 7: Switch the existing sweeps over to the harness**

`bench/construct_sweep.jl` and `bench/givensq_compaction.jl` each carry their own copy of the
recorder and the JSON writer, and neither checks the CPU governor. Both move to the shared one so
every sweep is guarded the same way.

Delete its local `ROWS`, `record`, `median` and the `meta`/`open`/`JSON.print` block at the end.
In their place, immediately after the `using` line:

```julia
include(joinpath(@__DIR__, "harness.jl"))
single_threaded!()
require_locked_clock()
```

Rename its `record(...)` calls to `record!(...)` and give each one `family = "construction"`; the
`routine`, `variant`, `eltype`, `n`, `s`, `bench` and `relerr` keywords are unchanged. Replace the
final block with:

```julia
save_results("construction")
```

`bench/givensq_compaction.jl` takes the guard lines only. Its rows are a per-shape crossover fit
rather than one timed call each, so they do not fit `record!`'s schema, and `test/givens_q.jl`
reads the constant back out of the file it writes; leave its `sample`, its `samples` key and its
writer alone. Give it the three lines above so it refuses an unlocked clock like the others, and
re-run it on the gate host in Task 6 so the numbers `docs/src/q_representations.md` quotes come
from the same machine as the rest.

Re-run both at their smoke sizes to confirm nothing broke:

```bash
ALLOW_UNLOCKED_CLOCK=1 julia --project=bench bench/construct_sweep.jl
ALLOW_UNLOCKED_CLOCK=1 julia --project=bench bench/givensq_compaction.jl
```

Expected: the construction sweep prints every row it printed before, in a wider format — the
columns are now `family`, `routine`, `variant`, `eltype`, `m`, `n`, `s`, median, bytes and
`relerr`, where `s` is the flush block size it sweeps and its `m` prints as `-`. Its JSON rows
carry the same keys as before plus `"family"`, `"m"` and `"median_bytes"`. The compaction sweep
prints and writes exactly what it did, and now refuses an unlocked clock without
`ALLOW_UNLOCKED_CLOCK`.

- [ ] **Step 8: Extend `bench/README.md`**

Add, above what is already there:

```markdown
## Reproducing from a fresh checkout

    julia --project=bench bench/setup.jl
    julia --project=bench bench/construct_sweep.jl
    julia --project=bench bench/updating_sweep.jl
    julia --project=bench bench/plot_results.jl

The first three write JSON under `bench/results/`. The fourth reads only those files and writes
the plots and the benchmarks page; it never runs a benchmark. A sweep refuses to run on a host
whose CPU governor is not `performance`.

`bench/` links `qrupdate-ng` through `QRupdatesFast`, which is GPL-3.0-or-later. That is why the
comparison lives in this environment and not in the package's.
```

- [ ] **Step 9: Commit**

```bash
git add bench/
git commit -m "Share the benchmark harness between the sweeps"
```

---

### Task 2: Cholesky updating cells

**Files:**
- Create: `bench/updating_sweep.jl`
- Modify: `bench/test_bench.jl`

**Interfaces:**
- Consumes: `record!`, `single_threaded!`, `require_locked_clock`, `save_results`; `UpdatableFactorizations.{UpdatableCholesky, lowrankupdate!, lowrankdowndate!, delete_column!, insert_column!, shift_columns!}`; `LinearAlgebra.{cholesky, lowrankupdate!, lowrankdowndate!}`; `UpdatableCholeskyFactorizations.{updatable_cholesky, add_column!, remove_column!}`
- Produces: `cholesky_cells(ns)`

- [ ] **Step 1: Write the failing check**

Append to `bench/test_bench.jl`:

```julia
@testset "cholesky cells" begin
    include(joinpath(@__DIR__, "updating_sweep.jl"))
    empty!(ROWS)
    cholesky_cells((24,); samples = 5)
    routines = unique(r["routine"] for r in ROWS)
    @test issetequal(
        routines,
        [
            "lowrankupdate!", "lowrankdowndate!", "delete_column!",
            "append_column!", "insert_column!", "shift_columns!",
        ]
    )
    @test all(r -> r["family"] == "cholesky", ROWS)
    @test all(r -> r["relerr"] < 1.0e-12, ROWS)
    @test any(r -> r["variant"] == "LinearAlgebra", ROWS)
    @test any(r -> r["variant"] == "UpdatableCholeskyFactorizations", ROWS)
    @test any(r -> r["variant"] == "recompute", ROWS)
    @test all(
        r -> r["median_bytes"] == 0,
        filter(r -> r["variant"] == "UpdatableFactorizations", ROWS)
    )
    empty!(ROWS)
end
```

`updating_sweep.jl` must therefore define its cell functions and do its work under a
`if abspath(PROGRAM_FILE) == @__FILE__` guard, so including it from the check defines the
functions without running the full sweep. It must also guard its own `include` of `harness.jl`,
since `test_bench.jl` has already included it into `Main` by this point.

- [ ] **Step 2: Run it to verify it fails**

```bash
julia --project=bench bench/test_bench.jl
```

Expected: FAIL — `bench/updating_sweep.jl` does not exist.

- [ ] **Step 3: Write the file header and the Cholesky cells**

`bench/updating_sweep.jl`:

```julia
# Every updating verb against the standard library, the prior-art packages and recomputing the
# factorization from scratch. Recomputing is the baseline that says whether updating pays.

using LinearAlgebra, Random, Chairmarks
using UpdatableFactorizations
import UpdatableCholeskyFactorizations
import UpdatableQRFactorizations
import QRupdate
import QRupdatesFast

# The harness defines `const ROWS`; including it twice into the same module would redefine that
# constant and rebind the array a caller is already collecting into.
isdefined(Main, :ROWS) || include(joinpath(@__DIR__, "harness.jl"))

# The comparison packages are reached through `import`, so their exports never enter this file's
# scope. Naming the module at every call site keeps it visible which package a verb belongs to,
# and keeps the file correct if a `using` is ever added: UpdatableCholeskyFactorizations and
# UpdatableQRFactorizations both export the names UpdatableCholesky and UpdatableQR.
const UF = UpdatableFactorizations
const UCF = UpdatableCholeskyFactorizations
const UQRF = UpdatableQRFactorizations

spd(n) = (B = randn(n, n); Matrix(Symmetric(B * B' + n * I)))

function cholesky_cells(ns; samples::Int = 100)
    for n in ns
        A = spd(n)
        C0 = cholesky(Symmetric(A, :L))
        mkF() = UF.UpdatableCholesky(cholesky(Symmetric(A, :L)))
        mkC() = cholesky(Symmetric(A, :L))
        mkU() = UCF.updatable_cholesky(A, 2n)

        # Rank-1 update. The standard library consumes its vector, so it is handed a copy.
        v = randn(n)
        target = A + v * v'
        relerr(F) = norm(Matrix(F) - target) / norm(A)
        F = mkF(); UF.lowrankupdate!(F, v)
        record!(;
            family = "cholesky", routine = "lowrankupdate!",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @be(mkF, F -> UF.lowrankupdate!(F, v), evals = 1, samples = samples),
            relerr = relerr(F),
        )
        C = mkC(); LinearAlgebra.lowrankupdate!(C, copy(v))
        record!(;
            family = "cholesky", routine = "lowrankupdate!", variant = "LinearAlgebra",
            eltype = Float64, n,
            bench = @be(
                mkC, C -> LinearAlgebra.lowrankupdate!(C, copy(v)), evals = 1, samples = samples
            ),
            relerr = relerr(C),
        )
        record!(;
            family = "cholesky", routine = "lowrankupdate!", variant = "recompute",
            eltype = Float64, n,
            bench = @be(cholesky(Symmetric(target, :L)), evals = 1, samples = samples),
            relerr = relerr(cholesky(Symmetric(target, :L))),
        )

        # Rank-1 downdate. The scaling keeps A - w*w' positive definite at every size.
        w = v ./ (2 * sqrt(n))
        dtarget = A - w * w'
        drelerr(F) = norm(Matrix(F) - dtarget) / norm(A)
        F = mkF(); UF.lowrankdowndate!(F, w)
        record!(;
            family = "cholesky", routine = "lowrankdowndate!",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @be(mkF, F -> UF.lowrankdowndate!(F, w), evals = 1, samples = samples),
            relerr = drelerr(F),
        )
        C = mkC(); LinearAlgebra.lowrankdowndate!(C, copy(w))
        record!(;
            family = "cholesky", routine = "lowrankdowndate!", variant = "LinearAlgebra",
            eltype = Float64, n,
            bench = @be(
                mkC, C -> LinearAlgebra.lowrankdowndate!(C, copy(w)), evals = 1,
                samples = samples
            ),
            relerr = drelerr(C),
        )
        record!(;
            family = "cholesky", routine = "lowrankdowndate!", variant = "recompute",
            eltype = Float64, n,
            bench = @be(cholesky(Symmetric(dtarget, :L)), evals = 1, samples = samples),
            relerr = drelerr(cholesky(Symmetric(dtarget, :L))),
        )

        # Symmetric deletion of index 2.
        keep = [i for i in 1:n if i != 2]
        del = A[keep, keep]
        F = mkF(); UF.delete_column!(F, 2)
        record!(;
            family = "cholesky", routine = "delete_column!",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @be(mkF, F -> UF.delete_column!(F, 2), evals = 1, samples = samples),
            relerr = norm(Matrix(F) - del) / norm(A),
        )
        U = mkU(); UCF.remove_column!(U, 2)
        record!(;
            family = "cholesky", routine = "delete_column!",
            variant = "UpdatableCholeskyFactorizations", eltype = Float64, n,
            bench = @be(mkU, U -> UCF.remove_column!(U, 2), evals = 1, samples = samples),
            relerr = norm(Matrix(U) - del) / norm(A),
        )
        record!(;
            family = "cholesky", routine = "delete_column!", variant = "recompute",
            eltype = Float64, n,
            bench = @be(cholesky(Symmetric(del, :L)), evals = 1, samples = samples),
            relerr = norm(Matrix(cholesky(Symmetric(del, :L))) - del) / norm(A),
        )

        # Appending an index at the end, and inserting one in the middle. `Ag` is the same matrix
        # in both cases; the middle case permutes it so the inserted index lands at position 2.
        Bg = randn(n + 1, n + 1)
        Ag = Matrix(Symmetric(Bg * Bg' + (n + 1) * I))
        x = Ag[:, n + 1]
        mkFg() = UF.UpdatableCholesky(cholesky(Symmetric(Ag[1:n, 1:n], :L)))
        mkUg() = UCF.updatable_cholesky(Ag[1:n, 1:n], 2n + 2)
        F = mkFg(); UF.insert_column!(F, n + 1, x)
        record!(;
            family = "cholesky", routine = "append_column!",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @be(mkFg, F -> UF.insert_column!(F, n + 1, x), evals = 1, samples = samples),
            relerr = norm(Matrix(F) - Ag) / norm(Ag),
        )
        U = mkUg(); UCF.add_column!(U, x)
        record!(;
            family = "cholesky", routine = "append_column!",
            variant = "UpdatableCholeskyFactorizations", eltype = Float64, n,
            bench = @be(mkUg, U -> UCF.add_column!(U, x), evals = 1, samples = samples),
            relerr = norm(Matrix(U) - Ag) / norm(Ag),
        )
        record!(;
            family = "cholesky", routine = "append_column!", variant = "recompute",
            eltype = Float64, n,
            bench = @be(cholesky(Symmetric(Ag, :L)), evals = 1, samples = samples),
            relerr = norm(Matrix(cholesky(Symmetric(Ag, :L))) - Ag) / norm(Ag),
        )

        into2 = [1; n + 1; 2:n]
        Ai = Ag[into2, into2]
        xi = x[[1; n + 1; 2:n]]
        F = mkFg(); UF.insert_column!(F, 2, xi)
        record!(;
            family = "cholesky", routine = "insert_column!",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @be(mkFg, F -> UF.insert_column!(F, 2, xi), evals = 1, samples = samples),
            relerr = norm(Matrix(F) - Ai) / norm(Ai),
        )
        record!(;
            family = "cholesky", routine = "insert_column!", variant = "recompute",
            eltype = Float64, n,
            bench = @be(cholesky(Symmetric(Ai, :L)), evals = 1, samples = samples),
            relerr = norm(Matrix(cholesky(Symmetric(Ai, :L))) - Ai) / norm(Ai),
        )

        # Moving index 1 to the end, the widest shift the size admits.
        p = [2:n; 1]
        sh = A[p, p]
        F = mkF(); UF.shift_columns!(F, 1, n)
        record!(;
            family = "cholesky", routine = "shift_columns!",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @be(mkF, F -> UF.shift_columns!(F, 1, n), evals = 1, samples = samples),
            relerr = norm(Matrix(F) - sh) / norm(A),
        )
        record!(;
            family = "cholesky", routine = "shift_columns!", variant = "recompute",
            eltype = Float64, n,
            bench = @be(cholesky(Symmetric(sh, :L)), evals = 1, samples = samples),
            relerr = norm(Matrix(cholesky(Symmetric(sh, :L))) - sh) / norm(A),
        )
    end
    return
end
```

- [ ] **Step 4: Run the check to verify it passes**

```bash
julia --project=bench bench/test_bench.jl
```

Expected: PASS. Every `relerr` is below 1e-12 and every `UpdatableFactorizations` row measures
zero bytes.

- [ ] **Step 5: Commit**

```bash
git add bench/updating_sweep.jl bench/test_bench.jl
git commit -m "Benchmark the Cholesky updating verbs"
```

---

### Task 3: LU updating cells

**Files:**
- Modify: `bench/updating_sweep.jl`, `bench/test_bench.jl`

**Interfaces:**
- Consumes: `record!`; `UpdatableFactorizations.{UpdatableLU, lowrankupdate!}`; `LinearAlgebra.{lu, NoPivot}`; `QRupdatesFast.{lu1up!, lup1up!}`
- Produces: `lu_cells(ns)`

- [ ] **Step 1: Write the failing check**

Append to `bench/test_bench.jl`:

```julia
@testset "lu cells" begin
    empty!(ROWS)
    lu_cells((24,); samples = 5)
    @test issetequal(
        unique(r["routine"] for r in ROWS),
        ["lowrankupdate! nopivot", "lowrankupdate! pivoted"]
    )
    @test all(r -> r["family"] == "lu", ROWS)
    # QRupdatesFast's pivoted routine reaches libqrupdate's BLAS calls. Without the LP64
    # forwarding in single_threaded! it returns a residual near one instead of near zero.
    @test all(r -> r["relerr"] < 1.0e-12, ROWS)
    @test any(r -> r["variant"] == "QRupdatesFast", ROWS)
    @test all(
        r -> r["median_bytes"] == 0,
        filter(r -> r["variant"] == "UpdatableFactorizations", ROWS)
    )
    empty!(ROWS)
end
```

- [ ] **Step 2: Run it to verify it fails**

Expected: FAIL — `lu_cells` not defined.

- [ ] **Step 3: Write the implementation**

Append to `bench/updating_sweep.jl`, before the `PROGRAM_FILE` guard:

```julia
function lu_cells(ns; samples::Int = 100)
    for n in ns
        A = Matrix(randn(n, n) + n * I)
        u = randn(n)
        v = randn(n)
        target = A + u * v'

        # Unpivoted. QRupdatesFast.lu1up! overwrites both factors and takes L as m x n and R as
        # n x n, so each sample is handed freshly materialized copies.
        Gn = lu(A, NoPivot())
        mkFn() = UF.UpdatableLU(lu(A, NoPivot()))
        mkQn() = (Matrix(Gn.L), Matrix(Gn.U))
        F = mkFn(); UF.lowrankupdate!(F, u, v)
        record!(;
            family = "lu", routine = "lowrankupdate! nopivot",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @be(mkFn, F -> UF.lowrankupdate!(F, u, v), evals = 1, samples = samples),
            relerr = norm(Matrix(F) - target) / norm(A),
        )
        Ln, Rn = mkQn(); QRupdatesFast.lu1up!(Ln, Rn, copy(u), copy(v))
        record!(;
            family = "lu", routine = "lowrankupdate! nopivot", variant = "QRupdatesFast",
            eltype = Float64, n,
            bench = @be(
                mkQn, t -> QRupdatesFast.lu1up!(t[1], t[2], copy(u), copy(v)), evals = 1,
                samples = samples
            ),
            relerr = norm(Ln * Rn - target) / norm(A),
        )
        record!(;
            family = "lu", routine = "lowrankupdate! nopivot", variant = "recompute",
            eltype = Float64, n,
            bench = @be(lu(target, NoPivot()), evals = 1, samples = samples),
            relerr = norm(Matrix(lu(target, NoPivot())) - target) / norm(A),
        )

        # Pivoted. lup1up!'s permutation argument is in/out: the identity that holds afterwards
        # uses the vector it returns, in the same convention as LinearAlgebra.lu.
        Gp = lu(A)
        mkFp() = UF.UpdatableLU(lu(A))
        mkQp() = (Matrix(Gp.L), Matrix(Gp.U), Int32.(Gp.p))
        F = mkFp(); UF.lowrankupdate!(F, u, v)
        record!(;
            family = "lu", routine = "lowrankupdate! pivoted",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @be(mkFp, F -> UF.lowrankupdate!(F, u, v), evals = 1, samples = samples),
            relerr = norm(Matrix(F) - target) / norm(A),
        )
        Lp, Rp, pp = mkQp(); QRupdatesFast.lup1up!(Lp, Rp, pp, copy(u), copy(v))
        record!(;
            family = "lu", routine = "lowrankupdate! pivoted", variant = "QRupdatesFast",
            eltype = Float64, n,
            bench = @be(
                mkQp, t -> QRupdatesFast.lup1up!(t[1], t[2], t[3], copy(u), copy(v)), evals = 1,
                samples = samples
            ),
            relerr = norm(Lp * Rp - target[Int.(pp), :]) / norm(A),
        )
        record!(;
            family = "lu", routine = "lowrankupdate! pivoted", variant = "recompute",
            eltype = Float64, n,
            bench = @be(lu(target), evals = 1, samples = samples),
            relerr = norm(Matrix(lu(target)) - target) / norm(A),
        )
    end
    return
end
```

- [ ] **Step 4: Run the check to verify it passes**

Expected: PASS. If any `QRupdatesFast` row shows a residual near one and the console carries
`Error: no BLAS/LAPACK library loaded for dtrsv_()`, `single_threaded!()` was not called before
the cells ran.

The 1e-12 bound in the check is a measured bound, not a guess: `A = randn(n, n) + n*I` at n = 24
is only weakly diagonally dominant, and Bennett's retained-permutation update amplifies a small
pivot. Print the worst case the seeded run actually reaches and record it here, so a later
regression is visible as a change rather than as a threshold argument:

```bash
julia --project=bench -e '
    using Random; Random.seed!(20260908)
    include("bench/updating_sweep.jl")
    single_threaded!(); empty!(ROWS); lu_cells((24,); samples = 5)
    println("worst lu relerr ", maximum(r["relerr"] for r in ROWS))
'
```

Write the number it prints into this step. If it is within a factor of ten of 1e-12, raise the
seed's matrix conditioning rather than the threshold — a benchmark whose residual gate is close
to firing is measuring an ill-conditioned problem, not the verb.

- [ ] **Step 5: Commit**

```bash
git add bench/updating_sweep.jl bench/test_bench.jl
git commit -m "Benchmark the LU rank-1 update"
```

---

### Task 4: QR updating cells

**Files:**
- Modify: `bench/updating_sweep.jl`, `bench/test_bench.jl`

**Interfaces:**
- Consumes: `record!`; `UpdatableFactorizations.{UpdatableQR, GivensQ, lowrankupdate!, insert_column!, delete_column!, shift_columns!, insert_row!, delete_row!}`; `UpdatableQRFactorizations.{UpdatableGivensQR, add_column!, remove_column!}`; `QRupdate.{qraddcol, qrdelcol, qraddrow}`; `QRupdatesFast.qrshift!`
- Produces: `qr_cells(sizes)`

- [ ] **Step 0: Confirm the QR signatures this task calls**

Every cell below is written against a spelling this task does not own. Confirm each one against
the shipped package before writing any of them, from an MCP Julia session:

```julia
using UpdatableFactorizations
const UF = UpdatableFactorizations
A = randn(9, 4)
F = UF.UpdatableQR(A)                       # default capacity
J = UF.UpdatableQR(qr(A), UF.GivensQ)       # the representation-selecting form
@assert size(Matrix(F)) == (9, 4)
@assert size(Matrix(J)) == (9, 4)
@assert UF.capacity(F) == (18, 8)
for f in (UF.lowrankupdate!, UF.insert_column!, UF.delete_column!, UF.shift_columns!,
          UF.insert_row!, UF.delete_row!)
    @assert !isempty(methods(f))
end
```

If the representation-selecting constructor is spelled differently, or `Matrix`, `insert_row!`,
`delete_row!` or the default capacity differ from what the Precondition block claims, **stop this
task** and reconcile the plan with the shipped API first. A `MethodError` discovered here costs a
minute; the same one discovered on the gate host costs a sweep.

- [ ] **Step 1: Write the failing check**

Append to `bench/test_bench.jl`:

```julia
@testset "qr cells" begin
    empty!(ROWS)
    qr_cells(((48, 12),); samples = 5)
    @test issetequal(
        unique(r["routine"] for r in ROWS),
        [
            "lowrankupdate!", "insert_column!", "delete_column!", "shift_columns!",
            "insert_row!", "delete_row!",
        ]
    )
    @test all(r -> r["family"] == "qr", ROWS)
    @test all(r -> r["m"] == 48 && r["n"] == 12, ROWS)
    @test all(r -> r["relerr"] < 1.0e-12, ROWS)
    # One row per cell in the case matrix: 3 + 5 + 5 + 4 + 3 + 2 for a single size. A dropped
    # cell changes this count, which the routine-set assertion above cannot see.
    @test length(ROWS) == 22
    # Each prior-art package is present. `all` over an empty generator is true, so the flag
    # assertions below say nothing unless the rows they filter for actually exist.
    @test any(r -> r["variant"] == "QRupdate", ROWS)
    @test any(r -> r["variant"] == "UpdatableQRFactorizations", ROWS)
    @test any(r -> r["variant"] == "QRupdatesFast", ROWS)
    @test any(r -> r["variant"] == "recompute", ROWS)
    # The two comparisons that do a different amount of work are flagged, not silently ranked.
    @test all(get(r, "r_only", false) for r in ROWS if r["variant"] == "QRupdate")
    @test all(
        get(r, "full_q", false) for r in ROWS if r["variant"] == "UpdatableQRFactorizations"
    )
    # The implicit-rotation Q covers the four column verbs; the two row verbs throw on it.
    givens = filter(r -> r["variant"] == "UpdatableFactorizations (GivensQ)", ROWS)
    @test issetequal(
        unique(r["routine"] for r in givens),
        ["lowrankupdate!", "insert_column!", "delete_column!", "shift_columns!"]
    )
    @test all(r -> get(r, "fresh_log", false), givens)
    @test all(
        r -> r["median_bytes"] == 0,
        filter(r -> r["variant"] == "UpdatableFactorizations", ROWS)
    )
    @test all(r -> r["median_bytes"] <= givens_alloc_bound(r["m"]), givens)
    empty!(ROWS)
end
```

The two representations are held to different allocation standards, because they allocate for
different reasons.

A `DenseQ` verb is allocation-free and is asserted at exactly zero. `UpdatableQR` is constructed
at its default capacity of `(2m, 2n)`, so no column or row verb at this size reaches a growth,
and every scratch vector the verbs use is a field of the factorization. A non-zero
`median_bytes` there means a growth fired inside the timed region, which is a defect in the verb.

A `GivensQ` verb allocates by design, so its rows carry a bound. It appends between `n` and `m`
rotations to three growing vectors, and applying the implicit base factor needs a small
workspace of its own. The bound is what those two costs can reach: three vectors of 8-byte
entries, growing by doubling to at most `m` entries each, plus the base factor's workspace.
Alongside `qrerr` in `bench/updating_sweep.jl`:

```julia
# A GivensQ verb pushes its rotations onto three growing vectors and applies the implicit base
# factor through a workspace of its own, so it allocates where a DenseQ verb does not. The
# bound is the log growth a single verb can cause plus that workspace; exceeding it means a
# workspace is being sized per m, or per call, rather than once.
givens_alloc_bound(m) = 48 * m + 512
```

Record the observed values in Step 4's expected output. If a row exceeds the bound, tighten the
verb rather than the bound.

- [ ] **Step 2: Run it to verify it fails**

Expected: FAIL — `qr_cells` not defined.

- [ ] **Step 3: Write the implementation**

Append to `bench/updating_sweep.jl`:

```julia
# Reconstruction residual of a thin QR against the matrix it should factor.
qrerr(Q, R, A) = norm(Q * R - A) / norm(A)

# A GivensQ verb pushes its rotations onto three growing vectors and applies the implicit base
# factor through a workspace of its own, so it allocates where a DenseQ verb does not. The bound
# is the log growth a single verb can cause plus that workspace; exceeding it means a workspace
# is being sized per m, or per call, rather than once.
givens_alloc_bound(m) = 48 * m + 512
qrerr(F::UF.UpdatableQR, A) = norm(Matrix(F) - A) / norm(A)

# QRupdate maintains R alone, from the normal equations, so its residual is measured on R'R.
rerr(R, A) = norm(R' * R - A' * A) / norm(A' * A)

const RONLY = Dict{String, Any}("r_only" => true)
const FULLQ = Dict{String, Any}("full_q" => true)
# Each sample rebuilds the factorization, so a GivensQ cell measures one verb against an empty
# rotation log. Apply cost rises as the log grows; that curve is the compaction sweep's.
const FRESHLOG = Dict{String, Any}("fresh_log" => true)

const GIVENS = "UpdatableFactorizations (GivensQ)"

function qr_cells(sizes; samples::Int = 100)
    for (m, n) in sizes
        A = randn(m, n)
        Q0 = Matrix(qr(A).Q)
        R0 = Matrix(qr(A).R)
        mkF() = UF.UpdatableQR(A)
        # insert_row! and delete_row! throw on the implicit-rotation Q, so it is benchmarked on
        # the four column verbs only. The constructor spelling is the one Step 0 confirmed.
        mkJ() = UF.UpdatableQR(qr(A), UF.GivensQ)

        # Rank-1 update.
        u = randn(m)
        v = randn(n)
        target = A + u * v'
        F = mkF(); UF.lowrankupdate!(F, u, v)
        record!(;
            family = "qr", routine = "lowrankupdate!", variant = "UpdatableFactorizations",
            eltype = Float64, m, n,
            bench = @be(mkF, F -> UF.lowrankupdate!(F, u, v), evals = 1, samples = samples),
            relerr = qrerr(F, target),
        )
        J = mkJ(); UF.lowrankupdate!(J, u, v)
        record!(;
            family = "qr", routine = "lowrankupdate!", variant = GIVENS, eltype = Float64, m, n,
            bench = @be(mkJ, J -> UF.lowrankupdate!(J, u, v), evals = 1, samples = samples),
            relerr = qrerr(J, target), extra = FRESHLOG,
        )
        record!(;
            family = "qr", routine = "lowrankupdate!", variant = "recompute", eltype = Float64,
            m, n,
            bench = @be(qr(target), evals = 1, samples = samples),
            relerr = qrerr(Matrix(qr(target).Q), Matrix(qr(target).R), target),
        )

        # Column insertion at the end.
        x = randn(m)
        wide = [A x]
        F = mkF(); UF.insert_column!(F, n + 1, x)
        record!(;
            family = "qr", routine = "insert_column!", variant = "UpdatableFactorizations",
            eltype = Float64, m, n,
            bench = @be(mkF, F -> UF.insert_column!(F, n + 1, x), evals = 1, samples = samples),
            relerr = qrerr(F, wide),
        )
        J = mkJ(); UF.insert_column!(J, n + 1, x)
        record!(;
            family = "qr", routine = "insert_column!", variant = GIVENS, eltype = Float64, m, n,
            bench = @be(mkJ, J -> UF.insert_column!(J, n + 1, x), evals = 1, samples = samples),
            relerr = qrerr(J, wide), extra = FRESHLOG,
        )
        mkG() = UQRF.UpdatableGivensQR(A, n + 1)
        G = mkG(); UQRF.add_column!(G, x)
        record!(;
            family = "qr", routine = "insert_column!", variant = "UpdatableQRFactorizations",
            eltype = Float64, m, n,
            bench = @be(mkG, G -> UQRF.add_column!(G, x), evals = 1, samples = samples),
            relerr = qrerr(Matrix(G.Q)[:, 1:size(G.R, 1)], G.R, wide), extra = FULLQ,
        )
        record!(;
            family = "qr", routine = "insert_column!", variant = "QRupdate", eltype = Float64,
            m, n,
            bench = @be(QRupdate.qraddcol(A, R0, x), evals = 1, samples = samples),
            relerr = rerr(QRupdate.qraddcol(A, R0, x), wide), extra = RONLY,
        )
        record!(;
            family = "qr", routine = "insert_column!", variant = "recompute", eltype = Float64,
            m, n,
            bench = @be(qr(wide), evals = 1, samples = samples),
            relerr = qrerr(Matrix(qr(wide).Q), Matrix(qr(wide).R), wide),
        )

        # Column deletion.
        keepc = [j for j in 1:n if j != 2]
        narrow = A[:, keepc]
        F = mkF(); UF.delete_column!(F, 2)
        record!(;
            family = "qr", routine = "delete_column!", variant = "UpdatableFactorizations",
            eltype = Float64, m, n,
            bench = @be(mkF, F -> UF.delete_column!(F, 2), evals = 1, samples = samples),
            relerr = qrerr(F, narrow),
        )
        J = mkJ(); UF.delete_column!(J, 2)
        record!(;
            family = "qr", routine = "delete_column!", variant = GIVENS, eltype = Float64, m, n,
            bench = @be(mkJ, J -> UF.delete_column!(J, 2), evals = 1, samples = samples),
            relerr = qrerr(J, narrow), extra = FRESHLOG,
        )
        mkG2() = UQRF.UpdatableGivensQR(A, n)
        G = mkG2(); UQRF.remove_column!(G, 2)
        record!(;
            family = "qr", routine = "delete_column!", variant = "UpdatableQRFactorizations",
            eltype = Float64, m, n,
            bench = @be(mkG2, G -> UQRF.remove_column!(G, 2), evals = 1, samples = samples),
            relerr = qrerr(Matrix(G.Q)[:, 1:size(G.R, 1)], G.R, narrow), extra = FULLQ,
        )
        record!(;
            family = "qr", routine = "delete_column!", variant = "QRupdate", eltype = Float64,
            m, n,
            bench = @be(QRupdate.qrdelcol(R0, 2), evals = 1, samples = samples),
            relerr = rerr(QRupdate.qrdelcol(R0, 2), narrow), extra = RONLY,
        )
        record!(;
            family = "qr", routine = "delete_column!", variant = "recompute", eltype = Float64,
            m, n,
            bench = @be(qr(narrow), evals = 1, samples = samples),
            relerr = qrerr(Matrix(qr(narrow).Q), Matrix(qr(narrow).R), narrow),
        )

        # Column shift: move column 1 to the last position.
        pc = [2:n; 1]
        shifted = A[:, pc]
        F = mkF(); UF.shift_columns!(F, 1, n)
        record!(;
            family = "qr", routine = "shift_columns!", variant = "UpdatableFactorizations",
            eltype = Float64, m, n,
            bench = @be(mkF, F -> UF.shift_columns!(F, 1, n), evals = 1, samples = samples),
            relerr = qrerr(F, shifted),
        )
        J = mkJ(); UF.shift_columns!(J, 1, n)
        record!(;
            family = "qr", routine = "shift_columns!", variant = GIVENS, eltype = Float64, m, n,
            bench = @be(mkJ, J -> UF.shift_columns!(J, 1, n), evals = 1, samples = samples),
            relerr = qrerr(J, shifted), extra = FRESHLOG,
        )
        mkQR() = (copy(Q0), copy(R0))
        Qs, Rs = mkQR(); QRupdatesFast.qrshift!(Qs, Rs, 1, n)
        record!(;
            family = "qr", routine = "shift_columns!", variant = "QRupdatesFast",
            eltype = Float64, m, n,
            bench = @be(
                mkQR, t -> QRupdatesFast.qrshift!(t[1], t[2], 1, n), evals = 1, samples = samples
            ),
            relerr = qrerr(Qs, Rs, shifted),
        )
        record!(;
            family = "qr", routine = "shift_columns!", variant = "recompute", eltype = Float64,
            m, n,
            bench = @be(qr(shifted), evals = 1, samples = samples),
            relerr = qrerr(Matrix(qr(shifted).Q), Matrix(qr(shifted).R), shifted),
        )

        # Row insertion at the end.
        y = randn(n)
        tall = [A; transpose(y)]
        F = mkF(); UF.insert_row!(F, m + 1, y)
        record!(;
            family = "qr", routine = "insert_row!", variant = "UpdatableFactorizations",
            eltype = Float64, m, n,
            bench = @be(mkF, F -> UF.insert_row!(F, m + 1, y), evals = 1, samples = samples),
            relerr = qrerr(F, tall),
        )
        record!(;
            family = "qr", routine = "insert_row!", variant = "QRupdate", eltype = Float64, m, n,
            bench = @be(QRupdate.qraddrow(R0, reshape(y, 1, n)), evals = 1, samples = samples),
            relerr = rerr(QRupdate.qraddrow(R0, reshape(y, 1, n)), tall), extra = RONLY,
        )
        record!(;
            family = "qr", routine = "insert_row!", variant = "recompute", eltype = Float64, m, n,
            bench = @be(qr(tall), evals = 1, samples = samples),
            relerr = qrerr(Matrix(qr(tall).Q), Matrix(qr(tall).R), tall),
        )

        # Row deletion. A random Gaussian A has no leverage-one row, so the default rtol never
        # fires here; the failure path is the test suite's, not the benchmark's.
        keepr = [i for i in 1:m if i != 2]
        short = A[keepr, :]
        F = mkF(); UF.delete_row!(F, 2)
        record!(;
            family = "qr", routine = "delete_row!", variant = "UpdatableFactorizations",
            eltype = Float64, m, n,
            bench = @be(mkF, F -> UF.delete_row!(F, 2), evals = 1, samples = samples),
            relerr = qrerr(F, short),
        )
        record!(;
            family = "qr", routine = "delete_row!", variant = "recompute", eltype = Float64, m, n,
            bench = @be(qr(short), evals = 1, samples = samples),
            relerr = qrerr(Matrix(qr(short).Q), Matrix(qr(short).R), short),
        )
    end
    return
end
```

- [ ] **Step 4: Run the check to verify it passes**

Expected: PASS. `UpdatableQRFactorizations`'s `G.Q` is `m x m`, so its residual is taken against
its leading `size(G.R, 1)` columns; a `DimensionMismatch` from `Matrix(G.Q) * G.R` means that
slice was dropped.

Record the `median_bytes` each `GivensQ` row reports here, next to the bound. They are the first
numbers anyone comparing a later gate run against this one has.

- [ ] **Step 5: Commit**

```bash
git add bench/updating_sweep.jl bench/test_bench.jl
git commit -m "Benchmark the QR updating verbs"
```

---

### Task 5: The sweep entry point and its smoke run

**Files:**
- Modify: `bench/updating_sweep.jl`

**Interfaces:**
- Consumes: `cholesky_cells`, `lu_cells`, `qr_cells`, `single_threaded!`, `require_locked_clock`, `save_results`
- Produces: `bench/results/updating-<host>.json`

- [ ] **Step 1: Add the entry point**

Append to `bench/updating_sweep.jl`:

```julia
if abspath(PROGRAM_FILE) == @__FILE__
    single_threaded!()
    require_locked_clock()
    Random.seed!(20260908)
    cholesky_cells((64, 128, 256, 512, 1024))
    lu_cells((64, 128, 256, 512, 1024))
    qr_cells(((256, 64), (512, 128), (1024, 256), (2048, 512)))
    save_results("updating")
end
```

- [ ] **Step 2: Smoke-run it on this host**

```bash
ALLOW_UNLOCKED_CLOCK=1 julia --project=bench bench/updating_sweep.jl
```

Expected: 198 rows print — Cholesky 16 per size over 5 sizes, LU 6 over 5, QR 22 over 4; every
`relerr` is below 1e-12; every `UpdatableFactorizations` row, dense-Q and `GivensQ` alike, reads
`0 B`; a file appears at `bench/results/updating-<host>.json`. These numbers are exploratory and
are not committed.

- [ ] **Step 3: Check the recorded file round-trips**

```bash
julia --project=bench -e '
    using JSON
    meta = JSON.parsefile("bench/results/updating-" * gethostname() * ".json")
    rows = meta["rows"]
    println(length(rows), " rows, governor ", meta["governor"], ", commit ", meta["commit"])
    println(maximum(r["relerr"] for r in rows))
    println(length(rows[1]["samples"]))
    @assert length(rows) == 198
    @assert all(r -> length(r["samples"]) == 100, rows)
'
```

Expected: 198 rows, the maximum `relerr` below 1e-12, and 100 samples per row. The row count is
asserted rather than eyeballed, so a silently dropped cell fails this step.

- [ ] **Step 4: Delete the exploratory file**

```bash
rm bench/results/updating-$(hostname).json
```

Only the gate host's file is committed.

- [ ] **Step 5: Commit**

```bash
git add bench/updating_sweep.jl
git commit -m "Add the updating sweep entry point"
```

---

### Task 6: The benchmark gate on a clock-locked host

**Files:**
- Create: `bench/results/updating-<host>.json`, `bench/results/construction-<host>.json`,
  `bench/results/givensq_compaction-<host>.json`
- Modify: `docs/src/q_representations.md`

**Interfaces:**
- Consumes: `bench/setup.jl`, `bench/updating_sweep.jl`, `bench/construct_sweep.jl`,
  `bench/givensq_compaction.jl`
- Produces: the committed datapoints every published number is drawn from

`neuromancer`'s governor is `powersave` and its numbers are never gate-authoritative. `galen`
(24 threads) and `wintermute` (12 threads) both report `performance`. Use one of them, and use
the same one for all three sweeps so the three files describe one machine.

- [ ] **Step 1: Sync the working tree to the gate host and verify it**

```bash
rsync -av --delete --exclude .git --exclude bench/Manifest.toml --exclude docs/build \
  /home/el_oso/Documents/claude/UpdatableFactorizations.jl/ \
  galen:~/Documents/claude/UpdatableFactorizations.jl/
ssh galen 'cd ~/Documents/claude/UpdatableFactorizations.jl && md5sum src/*.jl bench/*.jl'
md5sum src/*.jl bench/*.jl
```

Compare every checksum before running anything. A stale remote checkout silently reproduces the
previous run's numbers.

- [ ] **Step 2: Confirm the host's clock is locked**

```bash
ssh galen 'hostname; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; nproc'
```

Expected: `performance`. If it is anything else, stop: fix the host or use the other one. Do not
set `ALLOW_UNLOCKED_CLOCK`.

- [ ] **Step 3: Instantiate and run the three sweeps**

```bash
ssh galen 'cd ~/Documents/claude/UpdatableFactorizations.jl && julia --project=bench bench/setup.jl'
ssh galen 'cd ~/Documents/claude/UpdatableFactorizations.jl && julia --project=bench bench/updating_sweep.jl'
ssh galen 'cd ~/Documents/claude/UpdatableFactorizations.jl && julia --project=bench bench/construct_sweep.jl'
ssh galen 'cd ~/Documents/claude/UpdatableFactorizations.jl && julia --project=bench bench/givensq_compaction.jl'
```

All three sweeps are single-threaded by construction, so they leave the host's other cores free.
The compaction sweep runs here too: its writer now emits a host-suffixed file, and the three
result files must describe one machine.

- [ ] **Step 4: Copy the results back**

```bash
rsync -av galen:~/Documents/claude/UpdatableFactorizations.jl/bench/results/ \
  /home/el_oso/Documents/claude/UpdatableFactorizations.jl/bench/results/
```

- [ ] **Step 5: Check the gate**

```bash
julia --project=bench -e '
    using JSON, Printf
    meta = JSON.parsefile("bench/results/updating-galen.json")
    rows = meta["rows"]
    @assert meta["governor"] == "performance"
    @assert meta["blas_threads"] == 1
    @assert maximum(r["relerr"] for r in rows) < 1e-12
    @assert length(rows) == 198
    ours = filter(r -> startswith(r["variant"], "UpdatableFactorizations"), rows)
    dense = filter(r -> r["variant"] == "UpdatableFactorizations", ours)
    givens = filter(r -> r["variant"] == "UpdatableFactorizations (GivensQ)", ours)
    @assert all(r -> r["median_bytes"] == 0, dense)
    @assert all(r -> r["median_bytes"] <= 48 * r["m"] + 512, givens)

    # A recompute cell allocates its result and mutates nothing, so it is timed with a single
    # positional argument. Written with two, Chairmarks would treat the expression as `setup`,
    # run it outside the timing region, and record the cost of the second argument instead --
    # a few nanoseconds, which every published ratio then divides by.
    for family in unique(r["family"] for r in rows)
        here = filter(r -> r["family"] == family, rows)
        base = filter(r -> r["variant"] == "recompute", here)
        fastest = minimum(r["median_seconds"] for r in here)
        @assert minimum(r["median_seconds"] for r in base) >= fastest
        @assert minimum(r["median_seconds"] for r in base) > 1e-6
    end

    for r in ours
        base = only(filter(
            q -> q["family"] == r["family"] && q["routine"] == r["routine"] &&
                q["variant"] == "recompute" && q["n"] == r["n"], rows))
        @printf("%-8s %-24s %-34s n=%-5d  %.2fx recompute\n", r["family"], r["routine"],
                r["variant"], r["n"], base["median_seconds"] / r["median_seconds"])
    end
'
```

The gate passes when all five hold. It is an accuracy and reproducibility gate, not a speed gate:

1. The file records `performance` and one BLAS thread, and holds all 198 rows.
2. Every `relerr` is below 1e-12.
3. Every dense-Q row measures zero bytes, and every `UpdatableFactorizations (GivensQ)` row is
   within `48m + 512`. The two differ because the representations do: a dense-Q verb works
   entirely in storage the factorization already owns, while a `GivensQ` verb pushes between `n`
   and `m` rotations onto three growing vectors and applies the implicit base factor through a
   workspace of its own. Zero is unreachable for the second and would be a false gate; the bound
   is what its two costs can reach, and it still catches a workspace sized per call.
4. No `recompute` median is faster than the fastest updating median in its family, and the
   fastest `recompute` in a family exceeds a microsecond. Recomputing a factorization at the
   smallest size here is a matrix operation on 64 x 64 or larger; a recompute median in the
   nanoseconds means the cell timed the wrong thing, which no ratio check would show.
5. Every verb produces a ratio against recomputing, whatever that ratio is. A verb slower than
   recomputing at some size is a result to publish, not a failure to hide.

- [ ] **Step 6: Bring the Q representations page onto the gate host's run**

The compaction sweep writes `bench/results/givensq_compaction-<host>.json`, so this run replaces
the file for this host and leaves any other host's alone. Re-derive the compaction numbers
`docs/src/q_representations.md` quotes from the file this run wrote, and check that the page's
filename reference names it. The page and the benchmarks page then describe the same machine.

If a result file from an earlier host is still committed, delete it in this commit and check
that the `_COMPACT_ROTATIONS` item in `test/givens_q.jl`, which resolves the file by prefix and
asserts there is exactly one, still passes.

- [ ] **Step 7: Commit the datapoints**

```bash
git add bench/results docs/src/q_representations.md
git commit -m "Record the updating benchmarks on a clock-locked host"
```

---

### Task 7: Plots and the benchmarks page, generated from the saved data

**Files:**
- Create: `bench/plot_results.jl`, `docs/src/benchmarks.md`, `docs/src/assets/bench-cholesky.svg`, `docs/src/assets/bench-lu.svg`, `docs/src/assets/bench-qr.svg`
- Modify: `bench/test_bench.jl`

**Interfaces:**
- Consumes: `bench/results/updating-<host>.json`, `bench/results/construction-<host>.json`
- Produces: three SVGs and `docs/src/benchmarks.md`, both regenerated from the JSON alone

- [ ] **Step 1: Write the failing check**

Append to `bench/test_bench.jl`:

```julia
@testset "plot_results reads only the saved data" begin
    src = read(joinpath(@__DIR__, "plot_results.jl"), String)
    # The script must not construct a factorization or call a benchmark macro: the published
    # numbers come from the recorded file, not from a fresh measurement.
    for forbidden in ("@be", "@b ", "UpdatableCholesky(", "UpdatableLU(", "UpdatableQR(")
        @test !occursin(forbidden, src)
    end
    @test occursin("JSON.parsefile", src)
end
```

- [ ] **Step 2: Run it to verify it fails**

Expected: FAIL — `bench/plot_results.jl` does not exist.

- [ ] **Step 3: Write `bench/plot_results.jl`**

```julia
# Regenerates the plots and the benchmarks page from the recorded samples. Runs no benchmark: the
# numbers it publishes are exactly the numbers in bench/results.

using CairoMakie, JSON, Printf, Dates

CairoMakie.activate!(type = "svg")

const RESULTS = joinpath(@__DIR__, "results")
const ASSETS = joinpath(dirname(@__DIR__), "docs", "src", "assets")
const PAGE = joinpath(dirname(@__DIR__), "docs", "src", "benchmarks.md")

# Exactly one recorded file per sweep is committed, and it is the one published. Selecting by
# modification time would make the published page depend on checkout order, since git does not
# preserve mtimes.
function recorded(stem)
    files = filter(f -> startswith(f, stem * "-") && endswith(f, ".json"), readdir(RESULTS))
    isempty(files) && error("no $stem results in $RESULTS; run the sweep on a clock-locked host")
    length(files) == 1 || error(
        "$(length(files)) $stem files in $RESULTS ($(join(sort(files), ", "))); commit the gate " *
            "host's file and delete the exploratory ones"
    )
    path = joinpath(RESULTS, only(files))
    meta = JSON.parsefile(path)
    meta["governor"] == "performance" ||
        error("$path was recorded with governor \"$(meta["governor"])\" and cannot be published")
    return meta
end

# QR is swept over (m, n) pairs with m as the size axis; the other families are square.
xaxis(family) = family == "qr" ? "m" : "n"

function panel!(ax, cells, routine, key, colors)
    here = filter(r -> r["routine"] == routine, cells)
    for variant in sort(unique(r["variant"] for r in here))
        pts = sort(
            [(Float64(r[key]), r["median_seconds"]) for r in here if r["variant"] == variant];
            by = first
        )
        scatterlines!(ax, first.(pts), last.(pts); color = colors[variant], label = variant)
    end
    return ax
end

function figure(rows, family, title)
    cells = filter(r -> r["family"] == family, rows)
    routines = unique(r["routine"] for r in cells)
    key = xaxis(family)
    # One color per variant across the whole family, and one legend built from that map. Not
    # every panel carries every variant, so a per-axis color cycle would draw the same variant in
    # different colors in different panels, and a legend taken from one axis would omit the
    # variants that panel happens not to contain.
    variants = sort(unique(r["variant"] for r in cells))
    palette = CairoMakie.Makie.wong_colors()
    colors = Dict(v => palette[mod1(i, length(palette))] for (i, v) in enumerate(variants))
    ncols = min(3, length(routines))
    nrows = cld(length(routines), ncols)
    fig = Figure(size = (420 * ncols, 340 * nrows + 90))
    for (k, routine) in enumerate(routines)
        i, j = fldmod1(k, ncols)
        ax = Axis(
            fig[i, j]; title = routine, xscale = log10, yscale = log10,
            xlabel = key, ylabel = "seconds"
        )
        panel!(ax, cells, routine, key, colors)
    end
    Legend(
        fig[nrows + 1, 1:ncols],
        [MarkerElement(color = colors[v], marker = :circle) for v in variants], variants;
        orientation = :horizontal, nbanks = 2, framevisible = false
    )
    Label(fig[0, 1:ncols], title; fontsize = 18, font = :bold)
    path = joinpath(ASSETS, "bench-$family.svg")
    mkpath(ASSETS)
    save(path, fig)
    println("wrote $path")
    return path
end

function ratio_table(io, rows, family)
    key = xaxis(family)
    println(io, "| operation | $key | implementation | median | vs recomputing | notes |")
    println(io, "| --- | --- | --- | --- | --- | --- |")
    for routine in unique(r["routine"] for r in rows if r["family"] == family)
        cells = filter(r -> r["family"] == family && r["routine"] == routine, rows)
        for size in sort(unique(r[key] for r in cells))
            here = filter(r -> r[key] == size, cells)
            base = only(filter(r -> r["variant"] == "recompute", here))
            for r in here
                note = get(r, "r_only", false) ? "maintains R only" :
                    get(r, "full_q", false) ? "maintains a full m x m Q" :
                    get(r, "fresh_log", false) ? "empty rotation log" : ""
                @printf(
                    io, "| %s | %d | `%s` | %.3f ms | %.2fx | %s |\n",
                    routine, size, r["variant"], 1.0e3 * r["median_seconds"],
                    base["median_seconds"] / r["median_seconds"], note
                )
            end
        end
    end
    return
end

# The construction ratios quoted on this page are read out of the recorded construction sweep,
# like every other number here. Each one is a LAPACK baseline's median divided by this package's
# median for the same routine and size, so a value below 1.00 says the package is slower by that
# factor. `stdlib` rows are neither a LAPACK baseline nor this package's code.
function construction_span(meta)
    rows = meta["rows"]
    ours(routine) = filter(
        r -> r["routine"] == routine && r["eltype"] == "Float64" &&
            !startswith(r["variant"], "lapack") && !startswith(r["variant"], "stdlib"),
        rows
    )
    baseline(routine, n) = minimum(
        r["median_seconds"] for r in rows
            if r["routine"] == routine && r["n"] == n && startswith(r["variant"], "lapack")
    )
    ratios = Float64[]
    for routine in unique(r["routine"] for r in rows)
        cells = ours(routine)
        for n in unique(r["n"] for r in cells)
            best = minimum(r["median_seconds"] for r in cells if r["n"] == n)
            push!(ratios, baseline(routine, n) / best)
        end
    end
    isempty(ratios) && error("no construction rows to compare against a LAPACK baseline")
    return extrema(ratios)
end

preamble(lo, hi) = """
# Benchmarks

Every number on this page comes from a recorded sample file under `bench/results/`, written by
`bench/updating_sweep.jl` on a host whose CPU clock is locked to the `performance` governor, with
BLAS restricted to one thread. `bench/plot_results.jl` reads those files and writes this page and
its figures; it runs no benchmark of its own. To reproduce the whole thing from a checkout:

    julia --project=bench bench/setup.jl
    julia --project=bench bench/updating_sweep.jl
    julia --project=bench bench/plot_results.jl

## What is compared

`recompute` is the baseline: throw the factorization away and call `cholesky`, `lu` or `qr` again
on the modified matrix. It is what an updating routine has to beat to be worth using, and the
`vs recomputing` column is that ratio — above 1.00 means updating is faster than starting over at
that size.

Three comparisons do a different amount of work and are marked in the tables rather than ranked
against the others. `QRupdate` maintains `R` alone, from the normal equations, and never forms or
updates `Q`. `UpdatableQRFactorizations` maintains a full `m x m` `Q` where this package maintains
a thin `m x n` one. And every `GivensQ` cell starts from a freshly built factorization, so it
shows one verb against an empty rotation log; apply cost rises as the log grows, which is what
the compaction policy on [Q representations](@ref) is for.

Building a factorization from scratch goes through `LinearAlgebra`, and this package offers no
faster way to do that: its own construction routines measure $(@sprintf("%.2f", lo)) to
$(@sprintf("%.2f", hi)) of the blocked LAPACK routine they are compared against, where 1.00 would
be parity. See [Construction](@ref) for those numbers.
"""

function main()
    meta = recorded("updating")
    lo, hi = construction_span(recorded("construction"))
    rows = meta["rows"]
    open(PAGE, "w") do io
        println(io, "<!-- Generated by bench/plot_results.jl from bench/results. Edit the")
        println(io, "     script, not this file. -->")
        println(io)
        println(io, preamble(lo, hi))
        @printf(
            io, "\nRecorded on `%s`, governor `%s`, Julia %s, %s, %d BLAS thread, %s, commit `%s`.\n",
            meta["host"], meta["governor"], meta["julia"], meta["blas"], meta["blas_threads"],
            meta["date"], meta["commit"]
        )
        for (family, title) in
                ("cholesky" => "Cholesky", "lu" => "LU", "qr" => "QR")
            any(r -> r["family"] == family, rows) || continue
            figure(rows, family, "$title updating")
            println(io, "\n## $title\n")
            println(io, "![$title updating](assets/bench-$family.svg)\n")
            ratio_table(io, rows, family)
        end
    end
    println("wrote $PAGE")
    return
end

main()
```

- [ ] **Step 4: Run it and check the output**

```bash
julia --project=bench bench/plot_results.jl
julia --project=bench bench/test_bench.jl
```

Expected: three SVGs under `docs/src/assets/`, a `docs/src/benchmarks.md` whose tables carry a
row for every cell in the JSON, and the check passes. Open the page and confirm four things: no
sentence claims the package is faster than LAPACK at building a factorization; the construction
ratios in the preamble match a hand computation from `construction-galen.json`; the QR figure's
x-axis reads `m` and its table's size column holds the QR row counts; and each figure's legend
lists every variant that appears anywhere in that family, with the colors the panels drew.

- [ ] **Step 5: Commit**

```bash
git add bench/plot_results.jl bench/test_bench.jl docs/src/benchmarks.md docs/src/assets
git commit -m "Generate the benchmark plots and page from the saved samples"
```

---

### Task 8: Documentation completion

**Files:**
- Modify: `docs/make.jl`, `docs/src/index.md`, `docs/src/provenance.md`, `README.md`

**Interfaces:**
- Consumes: every public name in the package
- Produces: the eight-page site of the spec's section 9

- [ ] **Step 1: Wire the eight pages into `docs/make.jl`**

```julia
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Construction" => "construction.md",
        "Updating and downdating" => "updating.md",
        "Q representations" => "q_representations.md",
        "Benchmarks" => "benchmarks.md",
        "Provenance" => "provenance.md",
        "API" => "api.md",
    ],
```

- [ ] **Step 2: Confirm all eight source files exist**

```bash
ls docs/src/{index,getting_started,construction,updating,q_representations,benchmarks,provenance,api}.md
```

`construction.md` is written alongside the construction routines and `q_representations.md`
alongside the Q representations. If either is missing, that milestone is incomplete and this task
stops here rather than inventing a page for code someone else owns.

- [ ] **Step 3: Bring `docs/src/index.md` up to what the package ships**

Replace the two stale sentences. The first paragraph loses "instead of recomputing the
factorization from scratch" as a bare claim and names what is actually maintained; the line
"QR support is planned but not yet implemented" goes.

```markdown
# UpdatableFactorizations.jl

UpdatableFactorizations.jl keeps a Cholesky, LU or QR factorization current as the matrix it
factors is modified. It supports rank-1 update and downdate, symmetric insertion, deletion and
shifting of indices for Cholesky, rank-1 update for LU, and column and row insertion, deletion
and shifting for QR.

Building a factorization from scratch goes through `LinearAlgebra` (`cholesky`, `lu`, `qr`): this
package does not offer a faster way to do that. Its value is in keeping an existing factorization
current under modification. Each update runs in `O(n^2)` operations, and does not allocate once a
factorization's scratch storage has been sized to the operation.

The Cholesky rank-1 update and downdate extend `LinearAlgebra.lowrankupdate!` and
`LinearAlgebra.lowrankdowndate!` with methods for `UpdatableCholesky`, rather than replacing the
existing methods those functions already provide for `Cholesky`.

See [Getting started](@ref) to construct a factorization, [Updating and downdating](@ref) for a
worked example of each operation, [Benchmarks](@ref) for how each verb compares against
recomputing and against the prior art, and [Provenance](@ref) for the article each algorithm
derives from and the reference implementations consulted.
```

- [ ] **Step 4: Extend `docs/src/provenance.md`**

Add to the reference-implementation table the row for the wrapper that was measured against, and
state what each MIT package was consulted for now that the benchmarks exist:

```markdown
| `QRupdate.jl` | MIT | API shape; benchmark comparison, maintaining `R` alone |
| `UpdatableQRFactorizations.jl` | MIT | API shape; benchmark comparison, maintaining a full `m x m` `Q` |
| `UpdatableCholeskyFactorizations.jl` | MIT | the capacity-with-active-block storage idea; benchmark comparison |
```

The closing paragraphs stay as they are: `qrupdate-ng` is GPL-3.0-or-later, its source was not
read, and `QRupdatesFast.jl` appears only in the benchmark environment.

- [ ] **Step 5: Update `README.md`**

Its opening paragraph carries the same bare claim `index.md` sheds in Step 3 — "without
recomputing the factorization from scratch" — and is rewritten the same way, naming what is
maintained instead of asserting a comparison the page does not measure:

```markdown
Keeps a Cholesky, LU or QR factorization current as the matrix it factors is
modified: rank-1 update and downdate, symmetric insertion, deletion and
shifting of indices for Cholesky, rank-1 update for LU, and column and row
insertion, deletion and shifting for QR. Each update runs in `O(n^2)`
operations and does not allocate.
```

Then add a Benchmarks line after the provenance line:

```markdown
Measured comparisons against `LinearAlgebra`, `QRupdate.jl`,
`UpdatableQRFactorizations.jl`, `UpdatableCholeskyFactorizations.jl`, the
`qrupdate-ng` wrapper `QRupdatesFast.jl`, and against recomputing the
factorization from scratch, are on the
[benchmarks page](https://el-oso.github.io/UpdatableFactorizations.jl/dev/benchmarks).
Building a factorization from scratch goes through `LinearAlgebra`; this
package does not offer a faster way to do that.
```

- [ ] **Step 6: Build the docs**

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Expected: builds with no missing-docstring warnings and no broken cross-references. `@ref`
targets `Construction`, `Benchmarks` and `Q representations` must all resolve.

- [ ] **Step 7: Commit**

```bash
git add docs README.md
git commit -m "Complete the documentation site"
```

---

### Task 9: Registration readiness

**Files:**
- Create: `.github/workflows/TagBot.yml`
- Modify: `Project.toml` if the audit finds a gap

**Interfaces:**
- Consumes: `Project.toml`, `test/Project.toml`, `LICENSE`
- Produces: a repository that satisfies General's automatic-merge conditions

- [ ] **Step 1: Audit the package environment**

From an MCP Julia session:

```julia
using Pkg, TOML
p = TOML.parsefile("Project.toml")
@assert p["name"] == "UpdatableFactorizations"
@assert p["uuid"] == "621fe788-1cd9-43f6-8fef-7c11cfb05286"
@assert p["version"] == "0.1.0"
missing_compat = setdiff(keys(p["deps"]), keys(p["compat"]))
@assert isempty(missing_compat) "no [compat] entry for $missing_compat"
@assert haskey(p["compat"], "julia")
@assert !haskey(p["deps"], "QRupdatesFast")
@assert !haskey(p["deps"], "QRupdate_ng_jll")
```

Expected: every assertion holds. `[compat]` must name `julia` and every entry in `[deps]`;
General refuses a package that omits either.

- [ ] **Step 2: Confirm every dependency is registered in General**

A dependency that is not in General cannot be installed from it, which is the most common reason
automatic merge fails and is not something a local install test catches. Resolve each one against
the reachable registries, standard libraries excepted:

```julia
regs = Pkg.Registry.reachable_registries()
std = keys(Pkg.Types.stdlibs())
registered(uuid) = any(haskey(reg.pkgs, Base.UUID(uuid)) for reg in regs)
for env in ("Project.toml", "test/Project.toml")
    for (name, uuid) in get(TOML.parsefile(env), "deps", Dict{String, Any}())
        Base.UUID(uuid) in std && continue
        @assert registered(uuid) "$name ($uuid), a dependency of $env, is in no reachable registry"
    end
end
```

Expected: no assertion fires. `TypeContracts`, and the `StrictMode` and `StrictModeTest` the test
environment uses, must all resolve; if one does not, registration waits on that package rather
than proceeding.

- [ ] **Step 3: Confirm no GPL package reaches the package or test environments**

```julia
Pkg.activate("."); Pkg.instantiate(); Pkg.status(; mode = Pkg.PKGMODE_MANIFEST)
Pkg.activate("test"); Pkg.instantiate(); Pkg.status(; mode = Pkg.PKGMODE_MANIFEST)
```

Expected: neither manifest contains `QRupdatesFast` or `QRupdate_ng_jll`, the two packages that
carry the GPL link.

- [ ] **Step 4: Confirm the license file is present and is MIT**

```bash
head -3 LICENSE
```

Expected: an MIT header naming the copyright holder and 2026.

- [ ] **Step 5: Add `.github/workflows/TagBot.yml`**

```yaml
name: TagBot
on:
  issue_comment:
    types: [created]
  workflow_dispatch:
    inputs:
      lookback:
        default: "3"
permissions:
  actions: read
  checks: read
  contents: write
  deployments: read
  issues: read
  discussions: read
  packages: read
  pages: read
  pull-requests: read
  repository-projects: read
  security-events: read
  statuses: read
jobs:
  TagBot:
    if: github.event_name == 'workflow_dispatch' || github.actor == 'JuliaTagBot'
    runs-on: ubuntu-latest
    steps:
      - uses: JuliaRegistries/TagBot@v1
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
```

- [ ] **Step 6: Run the whole suite warm, then cold**

```
julia_run_testitems over the package, max_workers = 4, no filter
```

Expected: every item passes.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: passes. This is the gate that catches a test dependency the warm runner had already
loaded from somewhere else.

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/TagBot.yml Project.toml
git commit -m "Add TagBot and finish the registration prerequisites"
```

---

### Task 10: Registration request

**Files:**
- Modify: none

**Interfaces:**
- Consumes: a green CI run on the default branch
- Produces: a General registry pull request

- [ ] **Step 1: Establish the remote, or stop**

The checkout has no remote — `git remote -v` prints nothing, which is also why `docs/make.jl`
passes an explicit `repo = Documenter.Remotes.GitHub(...)`. Registration needs a public
repository, so the remote has to exist before anything else in this task:

```bash
git remote -v
curl -sS -o /dev/null -w '%{http_code}\n' https://github.com
gh auth status
```

If `git remote -v` is empty, create the repository under the user's account and attach it, with
the user's agreement on the name and visibility first:

```bash
gh repo create el-oso/UpdatableFactorizations.jl --public --source=. --remote=origin
git remote -v
```

Stop this task if github.com is unreachable from this host, if `gh` is not authenticated, or if
the user has not agreed to the repository being created. Do not invent a remote URL, and do not
work around an unreachable host.

- [ ] **Step 2: Open the pull request, with the text approved first**

Draft the pull-request title and body, show both to the user, and post only after approval —
`gh pr create --fill` would publish a title and body to GitHub with no such gate:

```bash
git push -u origin m4-benchmarks-registration
# after approval of the exact text:
gh pr create --title "<approved title>" --body "<approved body>"
```

Wait for CI, Documentation and Coveralls to report green on the pull request.

- [ ] **Step 3: Merge and confirm the deployed docs**

After merge, confirm `https://el-oso.github.io/UpdatableFactorizations.jl/dev/benchmarks` renders
the page and its three figures. A 404 means `docs/make.jl` is calling `Documenter.deploydocs`
rather than `DocumenterVitepress.deploydocs`.

- [ ] **Step 4: Ask for approval of the exact registration comment**

Registration is a comment posted under the user's GitHub account. Show the user the exact text
and the exact commit it will be posted on, and wait for approval before posting anything:

```
@JuliaRegistrator register
```

- [ ] **Step 5: Post it only after approval, and watch the registry pull request**

The General pull request must satisfy automatic merge: new package, valid name, `[compat]` for
every dependency including `julia`, a public repository, and a passing install test. If a
condition fails, fix it, tag no version, and re-register from a new commit.

- [ ] **Step 6: Confirm the tag**

Once the registry pull request merges, TagBot creates `v0.1.0`. Confirm:

```bash
git fetch --tags && git tag --list 'v*'
```

---

### Task 11: Full-suite gate

**Files:**
- Modify: none expected

- [ ] **Step 1: Run the whole suite**

```
julia_run_testitems over the package, max_workers = 4, no filter
```

Expected: every item passes.

- [ ] **Step 2: Run the benchmark checks**

```bash
julia --project=bench bench/test_bench.jl
```

Expected: every assertion passes.

- [ ] **Step 3: Confirm the published page still matches the recorded data**

```bash
julia --project=bench bench/plot_results.jl
git diff --stat docs/src/benchmarks.md docs/src/assets
```

Expected: no diff, on this host and on any other checkout. The generator publishes the one
committed result file per sweep and errors if it finds several, so its output does not depend on
which file the filesystem calls newest. A diff means the page was hand-edited after generation,
and the edit is moved into `bench/plot_results.jl` instead.

- [ ] **Step 4: Format**

Run `runic` over `src/`, `test/` and `bench/`. Commit any reformatting separately.

- [ ] **Step 5: Check the diff's comments and prose**

Re-read every comment, docstring and page paragraph added by this milestone. Anything that
references this plan, a milestone, a task number, a review, or how the code came to be is a
defect; rewrite it to state what is true now. Then grep the documentation for a speed claim:

```bash
grep -rniE 'faster|speedup|quicker|beats|outperform' docs/src README.md bench/README.md
```

Expected: every hit is read, not counted. The generated page says "above 1.00 means updating is
faster than starting over at that size", which is the updating-versus-recomputing comparison and
stays. A hit is a defect when it compares this package against LAPACK, OpenBLAS or the standard
library's blocked factorizations, or when it claims a speed win the recorded data does not carry.
Fix a defect in `bench/plot_results.jl` when it is on the generated page, and in the page source
otherwise.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Format and finish the benchmarks and documentation"
```

---

## Self-review

**Spec coverage for this milestone.** R12 (Tasks 7 and 8), R13 (Tasks 1–7), and the registration
gate of section 8 (Tasks 9 and 10). R15's provenance page is extended rather than rebuilt
(Task 8, step 4). R1's "no GPL in the dependency graph outside `bench/`" is asserted mechanically
in Task 9, step 3, rather than assumed, and the spec's condition that every dependency be
registered before this package is — section 8 — is asserted in Task 9, step 2.

**What the benchmark matrix deliberately does not cover.**

- Element types other than `Float64`. `QRupdatesFast` wraps `Float32`/`Float64`/`ComplexF32`/
  `ComplexF64` only, and `QRupdate` and `UpdatableQRFactorizations` are `Float64` in practice. A
  `BigFloat` or `Dual` column would have no comparison to draw, and element-type coverage is
  already the test suite's job.
- `GivensQ` on `insert_row!` and `delete_row!`. Both throw on that representation, which is a
  documented limit rather than a slow path, so there is nothing to time.
- How a `GivensQ` factorization behaves over a long run of updates. Each cell here rebuilds the
  factorization, so it measures a verb against an empty rotation log. The cost of a growing log
  and of compacting it is measured by the compaction sweep and published beside the policy it
  justifies.
- Multi-column and rank-k modifications. The package ships rank-1 verbs; there is nothing to
  measure.
- `chinx`, `chdex` and `chshx` from `qrupdate-ng`. `QRupdatesFast` 1.0.1 wraps three routines
  only — `qrshc!`, `lu1up!` and `lup1up!` — so the Cholesky verbs have no GPL comparison and the
  matrix says so rather than leaving a reader to wonder.

**Two allocation standards, one for each representation.** The dense-Q rows are held to exactly
zero bytes: those verbs work entirely in storage the factorization already owns, so anything
above zero is a growth firing inside the timed region. The `GivensQ` rows are held to `48m + 512`
instead, because that representation allocates by design — a verb pushes between `n` and `m`
rotations onto three growing vectors, and applying the implicit base factor needs a workspace.
Holding it to zero would be a gate that cannot pass; the bound still catches the regression that
matters, a workspace sized per call or per `m`.

**The two like-for-like caveats are in the data, not only in the prose.** `QRupdate` rows carry
`"r_only" => true` and `UpdatableQRFactorizations` rows carry `"full_q" => true`, so a future
reader of the JSON cannot rank them against a thin-Q update without seeing the flag. Both were
confirmed by reading the packages: `size(F.Q) == (40, 40)` for a 40 x 10 `UpdatableGivensQR`, and
`QRupdate`'s three exported routines take and return `R` alone.

**The failure mode this plan is built around.** `libqrupdate` resolves its BLAS symbols through
libblastrampoline's LP64 slot, which Julia leaves empty. Unresolved calls print an error line to
stderr and continue, so `lup1up!` and `qrshc!` return a plausible-looking factorization with a
residual near one. Nothing about the timing looks wrong. The defense is three-layered: the
forwarding lives in `single_threaded!` so every sweep gets it, every recorded row carries a
`relerr`, and the gate in Task 6 refuses a file whose maximum `relerr` exceeds 1e-12.

**Accuracy of the published ratios.** `median_seconds` is the median of 100 samples with
`evals = 1`, taken on a host locked to the `performance` governor with one BLAS thread. The
`vs recomputing` column divides the recompute median by the implementation's median at the same
size, so it is a ratio of two medians rather than a median of ratios; at these sample counts the
difference is below the run-to-run spread, and the raw samples are in the file for anyone who
wants a different statistic.
