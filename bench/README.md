# Construction benchmark

`construct_sweep.jl` measures `cholesky_crout`, `lu_crout`, and `qr_bcgs` against the
corresponding `LinearAlgebra` routine (`cholesky!`/LAPACK `potrf`, `lu!`/LAPACK `getrf`,
`qr!`/LAPACK `geqrf`), sweeping the matrix size `n` and the block size `s`. It also runs a
non-BLAS comparison at fixed size (`BigFloat` and `ForwardDiff.Dual` element types), where
`mul!` is itself a scalar loop rather than a BLAS call, to see whether blocking still helps.
Every sample from every benchmark is written out, not just the median, so tables and plots are
regenerated from the saved JSON rather than by re-running.

These algorithms are consistently slower than the LAPACK routines they are compared against.
The one routine that wins is `lu_crout` with `NoPivot()` against `LinearAlgebra.lu!(A,
NoPivot())`, because the standard library's unpivoted path is an unblocked generic fallback with
no blocked counterpart.

## Running it

The machine's CPU clock must be locked for the reported ratios to mean anything; an unpinned
clock makes absolute timings untrustworthy (ratios computed from samples interleaved within the
same `record` call still cancel drift, but comparisons *between* separate runs do not). From the
package root:

```sh
julia --project=bench bench/construct_sweep.jl
```

This activates the `bench` environment (`Chairmarks`, `JSON`, `ForwardDiff`, plus the standard
libraries the script uses), pins `BLAS.set_num_threads(1)`, and writes
`bench/results/construction-<hostname>.json`.

## Output

Each row of `"rows"` in the JSON records `routine`, `variant`, `eltype`, `n`, `s`, every sample
in seconds, the median, and a relative residual (`relerr`) against the matrix the factorization
was built from. Plots and tables are built from this file; do not re-run the sweep to regenerate
a plot, since the samples already on disk are what get plotted.
