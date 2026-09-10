## Reproducing from a fresh checkout

    julia --project=bench bench/setup.jl
    julia --project=bench bench/construct_sweep.jl
    julia --project=bench bench/updating_sweep.jl
    julia --project=bench bench/plot_results.jl

`setup.jl` develops this package into the `bench` environment and instantiates it, which is what
a checkout with no `bench/Manifest.toml` needs before any sweep can run. The next two scripts
write JSON under `bench/results/`. The fourth reads only those files and writes the plots and the
benchmarks page; it never runs a benchmark.

`bench/harness.jl` holds the machinery a sweep shares with the others: a JSON writer that keeps
every sample rather than a summary, a single-threaded/LP64 BLAS setup, a CPU-governor reader
recorded into each result file's metadata for provenance (nothing checks its value), and
`@mutating_bench`/`@pure_bench`, two thin wrappers around `Chairmarks.@be` that always time with
`evals = 1` and reject a stray positional argument rather than silently timing the wrong
expression.

`bench/` links `qrupdate-ng` through `QRupdatesFast`, which is GPL-3.0-or-later. That is why the
comparison lives in this environment and not in the package's.

# Construction benchmark

`construct_sweep.jl` measures `cholesky_crout`, `lu_crout`, and `qr_bcgs` against the
corresponding `LinearAlgebra` routine (`cholesky!`/LAPACK `potrf`, `lu!`/LAPACK `getrf`,
`qr!`/LAPACK `geqrf`), sweeping the matrix size `n` and the block size `s`. It also runs a
non-BLAS comparison at fixed size (`BigFloat` and `ForwardDiff.Dual` element types), where
`mul!` is itself a scalar loop rather than a BLAS call, to see whether blocking still helps.

Every comparison alternates single calls of its two sides in one loop (`paired` in the script):
one call to the baseline, one to the candidate, repeated for `ROUNDS` rounds, after one untimed
warmup call to each that pays for compilation. A clock or thermal drift over the run then moves
both sides of a given round together, so the ratio computed round by round cancels it; a ratio
of two independently measured medians would not, since the two sides would be measured minutes
apart. Every round is written out, not just the median, so tables and plots are regenerated from
the saved JSON rather than by re-running.

Each round is timed with `@timed`, which records two numbers rather than one:

- **wall** — the full elapsed time, garbage-collection pauses included. This is what a caller
  of the routine actually pays.
- **gc-net** — wall time minus the time spent in GC during that call. This is what answers
  "which routine is faster": the candidate allocates far more than the LAPACK baseline it is
  compared against, so a GC pause tripped during one side's call inflates that side's wall
  time without saying anything about which routine does more or less work. A pause landing on
  the cheaper side by chance can make the wall ratio favor the more expensive one for that
  round.

`paired` also runs `GC.gc()` once before each round, which starts every round from a comparable
heap. Measured on the `cholesky` `rankk` cell at `n=2000, s=64`, this brings the wall-ratio
spread down to match the gc-net spread; without it, the wall ratio swings across values the
gc-net ratio never approaches (a wall ratio of 1.7x with samples from 0.8x to 1.8x, against a
stable 1.1x once GC time is subtracted). Neither wall nor gc-net replaces the other, so both are
recorded, per round, for both sides.

These algorithms are generally slower than the LAPACK routines they are compared against, in
line with the standalone prototypes at `docs/superpowers/specs/alg1.jl`/`alg23.jl` and the
measurements in `docs/superpowers/specs/RESULTS.md`: roughly 0.6-0.95x of blocked LAPACK's speed,
falling as `s` grows. The one routine that wins is `lu_crout` with `NoPivot()` against
`LinearAlgebra.lu!(A, NoPivot())`, because the standard library's unpivoted path is an unblocked
generic fallback with no blocked counterpart; the prototype measured this at roughly 15-18x.

## Running it

From the package root:

```sh
julia --project=bench bench/construct_sweep.jl
```

This activates the `bench` environment (`JSON`, `ForwardDiff`, plus the standard libraries the
script uses), pins `BLAS.set_num_threads(1)`, and writes
`bench/results/construction.json`, replacing whatever was there. One file per sweep is committed
and it is the one the published tables and plots are built from. At the committed sizes (`n` up
to 4000, every `s`, plus the non-BLAS cells) and `ROUNDS = 12`, a full run takes on the order of
30-45 minutes.

## Output

Each row of `"rows"` in the JSON is one comparison: `routine`, `baseline`, `variant` (the
candidate), `eltype`, `n`, `s`, and the raw per-round samples and medians for both sides and
both timings — `base_wall_seconds`/`cand_wall_seconds`, `base_gcnet_seconds`/
`cand_gcnet_seconds`, and their `..._median_seconds` counterparts. The per-round ratio
(candidate over baseline) is recorded separately for each timing: `ratio_wall_samples` with
`ratio_wall_median`/`ratio_wall_min`/`ratio_wall_max`, and `ratio_gcnet_samples` with the same
three for gc-net. A relative residual (`relerr`) checks the candidate's factorization against
the matrix it was built from. `"rounds"` at the top level of the JSON records how many timed
rounds each comparison used. Plots and tables are built from this file; do not re-run the sweep
to regenerate a plot, since the samples already on disk are what get plotted.

One row is timing-only by construction: `lu` `crout nopivot` compared against `crout
rowmaximum` on the plain random fixture (`matrix => "random"`) has no accuracy bound to meet,
because running the unpivoted algorithm without pivoting on a matrix that was not built to need
none has no stability guarantee. Its `relerr` is recorded for completeness, not as a quality
figure.
