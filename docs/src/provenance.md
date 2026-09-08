# Provenance

The package is clean-room with respect to GPL prior art. Every routine it implements is derived
from a published article, listed below, and each algorithm's docstring cites the same source.

## Routines and their source articles

| routine | article |
| --- | --- |
| Cholesky rank-1 update | Gill, Golub, Murray and Saunders, *Methods for modifying matrix factorizations*, Mathematics of Computation 28 (1974), 505-535 |
| Cholesky rank-1 downdate | Bojanczyk, Brent, Van Dooren and de Hoog, *A note on downdating the Cholesky factorization*, SIAM Journal on Scientific and Statistical Computing 8 (1987), 210-221 |
| Cholesky symmetric delete and shift | Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5 |
| Cholesky symmetric insert | Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating the Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772-795 |
| LU rank-1 update, unpivoted | Bennett, *Triangular factors of modified matrices*, Numerische Mathematik 7 (1965), 217-221 |
| LU rank-1 update, pivoted | Stange, Griewank and Bollhöfer, *On the efficient update of rectangular LU-factorizations subject to low rank modifications*, ETNA 26 (2007), 161-177 |

## Reference implementations consulted

| implementation | license | consulted for |
| --- | --- | --- |
| `LinearAlgebra` (Julia standard library) | MIT | the `lowrankupdate!`/`lowrankdowndate!` signatures this package extends |
| `QRupdate.jl` | MIT | API shape |
| `UpdatableQRFactorizations.jl` | MIT | API shape |
| `UpdatableCholeskyFactorizations.jl` | MIT | the capacity-with-active-block storage idea |

`qrupdate-ng` is GPL-3.0-or-later. Its source was not read. It serves only as a benchmark target
and as a reference list of routine names.

`QRupdatesFast.jl` links `qrupdate-ng` and so may appear only in a benchmark environment, never
as a dependency of this package.
