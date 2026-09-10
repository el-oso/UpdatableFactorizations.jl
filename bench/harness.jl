# Shared machinery for the benchmark sweeps: every sample is kept, not just a summary, so the
# plots and the published tables are regenerated from the saved file rather than by re-running.

using LinearAlgebra, Chairmarks, JSON, Dates, Printf
import OpenBLAS32_jll

const ROWS = Dict{String, Any}[]

median(v) = sort(v)[(length(v) + 1) ÷ 2]

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

# Recorded into every result file's metadata (see `save_results`) so a reader knows what
# produced a given number. Not a gate: a sweep runs regardless of what this returns.
function governor()
    path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    return isfile(path) ? String(strip(read(path, String))) : "unknown"
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
    # `m` and `s` are family-specific -- `m` is the row count of a QR problem, `s` the flush block
    # size of a construction sweep -- and print as "-" for a family that has no such axis.
    @printf(
        "%-10s %-22s %-32s %-8s m=%-6s n=%-6d s=%-5s %9.3f ms  %8.0f B  relerr %.1e\n",
        family, routine, variant, string(eltype), isnothing(m) ? "-" : string(m), n,
        isnothing(s) ? "-" : string(s),
        1.0e3 * row["median_seconds"], row["median_bytes"], row["relerr"]
    )
    return row
end

# `@be`'s positional pipeline is `[[init] setup] f [teardown]`, resolved by argument count: a
# third positional argument is `teardown`, not `f`, so a mistyped cell can time the wrong
# expression without erroring. These two macros fix the positional shape so that mistake becomes
# a load-time error instead of a silently wrong number. `@mutating_bench` accepts exactly `setup`
# and `f` and hardcodes `evals = 1`, which a mutating verb needs on every call -- without it,
# Chairmarks reapplies the verb to the factorization `setup` already produced and already
# mutated. `@pure_bench` accepts exactly one self-contained expression, for a cell that allocates
# its own result and mutates nothing. Both reject a further bare positional argument; only
# trailing `key = value` options (`samples = ...`) are allowed through to `@be`.
function _reject_bare_positional(macroname, args)
    for a in args
        a isa Expr && a.head === :(=) ||
            error(
            "$macroname: extra arguments after the timed expression must be `key = value` " *
                "options (e.g. `samples = 100`); got `$a`, which `@be` would resolve as a " *
                "positional argument instead of timing it"
        )
    end
    return args
end

# `@be` tells a bare function symbol (call it) apart from a general expression (evaluate it) by
# inspecting the raw argument syntax, so escaping the pieces individually before forwarding them
# would hide that shape behind `:escape` nodes. Escaping the whole assembled call instead leaves
# every piece exactly as the caller wrote it, and still resolves it in the caller's scope.
macro mutating_bench(setup, f, opts...)
    _reject_bare_positional(:mutating_bench, opts)
    return esc(:(Chairmarks.@be($setup, $f, evals = 1, $(opts...))))
end

macro pure_bench(expr, opts...)
    _reject_bare_positional(:pure_bench, opts)
    return esc(:(Chairmarks.@be($expr, evals = 1, $(opts...))))
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
