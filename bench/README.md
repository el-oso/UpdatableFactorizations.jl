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
`bench/results/construction-<hostname>.json`. The hostname in the filename records where a run
came from; it is provenance, not a claim that any one machine's numbers are authoritative. At
the committed sizes (`n` up to 4000, every `s`, plus the non-BLAS cells) and `ROUNDS = 12`, a
full run takes on the order of 30-45 minutes.

## Output

Each row of `"rows"` in the JSON is one comparison: `routine`, `baseline`, `variant` (the
candidate), `eltype`, `n`, `s`, the raw per-round times for both sides (`base_samples`,
`cand_samples`) and their medians, the per-round ratio (`ratio_samples`, candidate over
baseline) with its median and its min/max spread, and a relative residual (`relerr`) for the
candidate's factorization against the matrix it was built from. `"rounds"` at the top level of
the JSON records how many timed rounds each comparison used. Plots and tables are built from
this file; do not re-run the sweep to regenerate a plot, since the samples already on disk are
what get plotted.

One row is timing-only by construction: `lu` `crout nopivot` compared against `crout
rowmaximum` on the plain random fixture (`matrix => "random"`) has no accuracy bound to meet,
because running the unpivoted algorithm without pivoting on a matrix that was not built to need
none has no stability guarantee. Its `relerr` is recorded for completeness, not as a quality
figure.
