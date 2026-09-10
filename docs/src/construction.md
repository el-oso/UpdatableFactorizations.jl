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

## Speed, relative to LAPACK

Every ratio below is candidate speed divided by baseline speed: **above 1.00 is a win, below
1.00 is a loss.** This is the opposite of the raw numbers stored in
`bench/results/construction-neuromancer.json`, which record candidate time divided by baseline
time — a ratio above 1.00 there means the candidate took *longer*. Figures here are single-BLAS-
thread medians from that file.

| routine | flush / setting | baseline | n=2000 | n=4000 |
| --- | --- | --- | --- | --- |
| `cholesky_crout` | symmetric rank-k, the default | `potrf` | 0.75 – 0.92 | 0.86 – 0.92 |
| `cholesky_crout` | full-block multiply | `potrf` | 0.51 – 0.56 | 0.51 – 0.52 |
| `lu_crout` | pivoted, the default | `getrf` | 0.79 – 0.90 | 0.83 – 0.89 |
| `lu_crout` | unpivoted | `getrf` | 0.87 – 0.97 | 0.89 – 0.96 |
| `lu_crout` | unpivoted | stdlib `lu!(A, NoPivot())` | 12.6× – 15.6× | 15.1× – 16.4× |
| `qr_bcgs` | `reorth = false` | `geqrf` plus forming thin `Q` | **1.47 – 1.61** | **1.38 – 1.62** |
| `qr_bcgs` | `reorth = true`, the default | `geqrf` plus forming thin `Q` | 0.75 – 0.84 | 0.72 – 0.84 |

Ranges span block sizes `s = 64, 128, 256` for Cholesky and LU and `s = 32, 64, 128, 256` for QR.

`qr_bcgs` with `reorth = false` is the one routine here faster than LAPACK. It builds `R` from
the projection coefficients the orthogonalization already produces, rather than closing with
`Q'*A`, which cost a further `2*m*n^2` and was 44% of the routine. `reorth = true`, the default,
runs that projection a second time and is still slower than the baseline; see
[Accuracy](@ref construction-accuracy) for what the setting buys.

Every other cell loses to LAPACK. The unpivoted `lu_crout` figures against stdlib's own
`NoPivot()` path are large only because that path is an unblocked generic fallback, not a LAPACK
call — measured against `getrf`, which is what a caller actually gets from `LinearAlgebra.lu`,
unpivoted `lu_crout` is 0.87 to 0.97, slightly slower like everything else in this table.
Substituting a full-block multiply for Cholesky's default symmetric rank-k flush costs a further
1.7 to 2.0 times.

## [Accuracy](@id construction-accuracy)

`cholesky_crout` reaches `norm(L*L' - A)/norm(A)` of 1.7e-16 to 2.4e-16 on a random SPD matrix, in
`potrf`'s own range (1.5e-16 to 1.8e-16 on the same matrices).

`lu_crout`'s accuracy depends heavily on which matrix it is measured against. On a diagonally
dominant matrix, where every leading principal minor is nonzero and partial pivoting never
swaps a row, unpivoted `lu_crout` reaches `norm(L*U - A)/norm(A)` of 1.7e-16 to 2.2e-16. That
fixture says nothing about pivoted accuracy, because it never exercises a pivot.

On a plain random matrix, which swaps rows at essentially every column, `LinearAlgebra.lu`
itself (via `getrf`) reaches a residual of 1.1e-14 at n=2000 and 2.2e-14 at n=4000; pivoted
`lu_crout` reaches 1.4e-14 and 2.6e-14 on the identical matrix, about 1.2 times LAPACK's own
residual. Quoting the diagonally dominant figure next to that ratio would suggest a hundredfold
gap that isn't there; the two numbers describe different matrices.

`qr_bcgs` trades one residual for the other, not both at once:

| quantity | `geqrf` + form `Q` | `qr_bcgs`, `reorth = false` | `qr_bcgs`, `reorth = true` |
| --- | --- | --- | --- |
| `norm(Q*R - A)/norm(A)` | 1.2e-15 | 5.6e-16 | 6.1e-16 |
| `norm(Q'Q - I)` | 1.0e-13 | 1.7e-12 | 1.6e-12 |

Reconstruction is *better* than the baseline's regardless of `reorth`, because `R` is built from
the same coefficients that built `Q`: `Q*R` reassembles `A` from the pieces subtracted from it
during elimination, an identity that holds however orthogonal the finished `Q` turns out to be.
Orthogonality is the setting `reorth` actually controls, and block classical Gram-Schmidt loses
it as the square of the condition number regardless of that setting on a well-conditioned matrix.
`reorth = true` gives orthogonality of order `eps` once the condition number times `eps` is well
below one, at roughly twice the cost (see the speed table above). `qr_householder`, which calls
`LinearAlgebra.qr` and wraps the result, is unconditionally more accurate on orthogonality than
either setting.

## Non-BLAS element types

The deferred flush only pays when it can hand a block to a BLAS kernel. Measured against the
same routine run with `s >= n`, which never flushes and so is the unblocked left-looking
factorization:

| element type | n | `s = 16` | `s = 64` |
| --- | --- | --- | --- |
| `BigFloat` | 120 | 0.48 | 0.69 |
| `ForwardDiff.Dual` | 200 | 0.96 | 0.99 |

Blocking a `BigFloat` factorization is a loss at both block sizes; for `ForwardDiff.Dual` it is
roughly neutral. Neither element type is a BLAS type, so `rankk!`'s default falls back to a full
matrix multiply on both triangles instead of `syrk!`/`herk!`, and the panel bookkeeping the
blocking adds has nothing to pay for it.

For `ForwardDiff.Dual` specifically, `LinearAlgebra.cholesky`'s generic fallback is 3.12 times
faster than this package's unblocked path. Use the standard library for `Dual`.

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
`herk!` for complex ones. Passing a full-block multiply instead is the `gemm` row of the speed
table above.

## Failure

`cholesky_crout` throws `PosDefException(c)` at the first column with a non-positive pivot.
`lu_crout` throws `ZeroPivotException(c)`; with `pivot = NoPivot()` that means the leading
principal minor is singular, and `rtol` widens the test from exact zero to a threshold relative
to the column's norm. `qr_bcgs` throws when a column's projection collapses onto the columns
before it, which means the matrix is rank deficient; it takes the same `rtol`, and needs it,
because a rank-deficient input in floating point leaves rounding noise rather than an exact
zero.
