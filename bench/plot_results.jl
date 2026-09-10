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
    path = joinpath(RESULTS, stem * ".json")
    isfile(path) || error("no $stem results in $RESULTS; run the sweep first")
    return JSON.parsefile(path)
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

# The convention held throughout this page: every ratio is a reference time divided by
# UpdatableFactorizations' time (or, for a prior-art row, that package's own time), so a value
# above 1.00 means the row runs faster than the reference and a value below 1.00 means it runs
# slower. The reference for an updating verb is `recompute`: throwing the factorization away and
# calling `cholesky`, `lu` or `qr` again.
speedup(base, cand) = base["median_seconds"] / cand["median_seconds"]

# Whether UpdatableFactorizations' own verb beats recomputing, size by size, for one routine.
# `nothing` values (a routine measured only at some sizes) never occur here -- every routine in
# the case matrix is measured at every size in its family -- so every ratio compares.
function verdict_row(rows, family, routine)
    key = xaxis(family)
    cells = filter(r -> r["family"] == family && r["routine"] == routine, rows)
    ratios = Pair{Int, Float64}[]
    for size in sort(unique(r[key] for r in cells))
        here = filter(r -> r[key] == size, cells)
        base = only(filter(r -> r["variant"] == "recompute", here))
        ours = only(filter(r -> r["variant"] == "UpdatableFactorizations", here))
        push!(ratios, size => speedup(base, ours))
    end
    losses = [size for (size, ratio) in ratios if ratio < 1]
    verdict = if isempty(losses)
        "wins at every $key measured"
    elseif length(losses) == length(ratios)
        "loses at every $key measured"
    else
        "loses at $key = " * join(string.(losses), ", ")
    end
    lo, hi = extrema(last, ratios)
    return routine, lo, hi, verdict
end

# The fastest other implementation of the same routine, and how this package's verb compares to it
# size by size. A verb can beat recomputing everywhere and still trail a specialized library, so
# the two verdicts are reported side by side rather than leaving the second one to the plots.
function rival_verdict(rows, family, routine)
    key = xaxis(family)
    cells = filter(r -> r["family"] == family && r["routine"] == routine, rows)
    others = unique(
        r["variant"] for r in cells
            if !(r["variant"] in ("recompute", "UpdatableFactorizations"))
    )
    isempty(others) && return "-", "no other implementation measured"
    best = nothing
    for name in others
        ratios = Pair{Int, Float64}[]
        for size in sort(unique(r[key] for r in cells))
            here = filter(r -> r[key] == size, cells)
            theirs = filter(r -> r["variant"] == name, here)
            ours = only(filter(r -> r["variant"] == "UpdatableFactorizations", here))
            isempty(theirs) && continue
            push!(ratios, size => speedup(only(theirs), ours))
        end
        isempty(ratios) && continue
        # Rank by the worst showing against that rival, so the hardest comparison is the one told.
        worst = minimum(last, ratios)
        (isnothing(best) || worst < best[2]) && (best = (name, worst, ratios))
    end
    isnothing(best) && return "-", "no other implementation measured"
    name, _, ratios = best
    losses = [size for (size, ratio) in ratios if ratio < 1]
    lo, hi = extrema(last, ratios)
    # A comparison that maintains a different amount of the factorization carries that here too,
    # not only in the detail table: the ratio alone would read as a like-for-like result.
    caveat = if any(r -> r["variant"] == name && get(r, "r_only", false), cells)
        " (which maintains `R` only, never `Q`)"
    elseif any(r -> r["variant"] == name && get(r, "full_q", false), cells)
        " (which maintains a full `m x m` `Q`)"
    else
        ""
    end
    text = if isempty(losses)
        @sprintf("%.2fx-%.2fx faster than `%s`%s", lo, hi, name, caveat)
    elseif length(losses) == length(ratios)
        @sprintf("%.2fx-%.2fx of `%s`%s, slower at every %s", lo, hi, name, caveat, key)
    else
        @sprintf(
            "%.2fx-%.2fx of `%s`%s, slower at %s = %s", lo, hi, name, caveat, key,
            join(string.(losses), ", ")
        )
    end
    return name, text
end

function verdict_table(io, rows, family)
    println(io, "| routine | worst | best | vs recomputing | vs the fastest other implementation |")
    println(io, "| --- | --- | --- | --- | --- |")
    for routine in unique(r["routine"] for r in rows if r["family"] == family)
        routine, lo, hi, verdict = verdict_row(rows, family, routine)
        _, rival = rival_verdict(rows, family, routine)
        @printf(io, "| `%s` | %.2fx | %.2fx | %s | %s |\n", routine, lo, hi, verdict, rival)
    end
    return
end

# The construction sweep pairs each candidate against a baseline call and stores candidate-time
# divided by baseline-time per round (`bench/construct_sweep.jl`'s own "wall" ratio). This page's
# ratio convention is the opposite direction (reference divided by candidate, above 1.00 is a
# win), so `speed` below is the reciprocal of the stored median. Only a genuine LAPACK baseline
# counts: some rows compare two LAPACK-only costs against each other (forming Q after `geqrf`,
# say) and are candidate-labeled "lapack ...", not this package's code.
function construction_span(meta)
    rows = meta["rows"]
    ours = filter(
        r -> startswith(r["baseline"], "lapack") && !startswith(r["variant"], "lapack"), rows
    )
    isempty(ours) && error("no construction rows compare this package against a LAPACK baseline")
    speeds = [r["base_wall_median_seconds"] / r["cand_wall_median_seconds"] for r in ours]
    return extrema(speeds)
end

preamble(lo, hi) = """
# Benchmarks

Every number on this page comes from a recorded sample file under `bench/results/`, written by
`bench/updating_sweep.jl` with BLAS restricted to one thread. `bench/plot_results.jl` reads those
files and writes this page and its figures; it runs no benchmark of its own. To reproduce the
whole thing from a checkout:

    julia --project=bench bench/setup.jl
    julia --project=bench bench/updating_sweep.jl
    julia --project=bench bench/plot_results.jl

## What is compared, and which way the ratios point

Every ratio on this page is a reference time divided by the time of the row it sits next to:
**above 1.00 means that row is faster than the reference, below 1.00 means it is slower.** For
an updating verb the reference is `recompute` -- throwing the factorization away and calling
`cholesky`, `lu` or `qr` again on the modified matrix, in the "vs recomputing" column below. That
is the package's central claim: whether keeping a factorization current is worth it compared to
starting over. A verdict table precedes each family's detail table and states, per routine, the
range of that ratio across the sizes measured and at which sizes, if any, updating loses.

Two of the prior-art comparisons do a different amount of work than the row they sit next to, and
are marked wherever their numbers appear, not only here: `QRupdate` maintains `R` alone, from the
normal equations, and never forms or updates `Q`, so it does strictly less work.
`UpdatableQRFactorizations` maintains a full `m x m` `Q` where this package maintains a thin
`m x n` one, so it does strictly more work.

Building a factorization from scratch goes through `LinearAlgebra`. This package's own
construction routines run from $(@sprintf("%.2f", lo))x to $(@sprintf("%.2f", hi))x the speed of
the blocked LAPACK routine they are compared against, where 1.00 is parity: most of those cells
are slower and one is faster. See [Construction](@ref) for which routine and why.
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
            io, "Recorded with governor `%s`, Julia %s, %s, %d BLAS thread, %s, commit `%s`.\n",
            meta["governor"], meta["julia"], meta["blas"], meta["blas_threads"],
            meta["date"], meta["commit"]
        )
        for (family, title) in ("cholesky" => "Cholesky", "lu" => "LU", "qr" => "QR")
            any(r -> r["family"] == family, rows) || continue
            figure(rows, family, "$title updating")
            println(io, "\n## $title\n")
            println(io, "![$title updating](assets/bench-$family.svg)\n")
            verdict_table(io, rows, family)
        end
    end
    println("wrote $PAGE")
    return
end

main()
