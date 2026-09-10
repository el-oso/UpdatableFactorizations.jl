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
| QR rank-1 update | Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating the Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772-795; Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5 |
| QR column insertion | Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating the Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772-795 |
| QR column deletion | Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5 |
| QR column shift | Reichel and Gragg, *Algorithm 686: FORTRAN subroutines for updating the QR decomposition*, ACM Transactions on Mathematical Software 16 (1990), 369-377 |
| QR row insertion | Golub and Van Loan, *Matrix Computations*, 4th edition, section 6.5 |
| QR row deletion | Daniel, Gragg, Kaufman and Stewart, *Reorthogonalization and stable algorithms for updating the Gram-Schmidt QR factorization*, Mathematics of Computation 30 (1976), 772-795; Reichel and Gragg, *Algorithm 686: FORTRAN subroutines for updating the QR decomposition*, ACM Transactions on Mathematical Software 16 (1990), 369-377 |
| Householder QR construction | `LinearAlgebra.qr`, which `qr_householder` calls; no algorithm is implemented here |
| Blocked Crout Cholesky | Camarero, *Simple, Fast and Practicable Algorithms for Cholesky, LU and QR Decomposition Using Fast Rectangular Matrix Multiplication*, arXiv:1812.02056 (2018), Algorithm 1 |
| Blocked Crout LU | Camarero, arXiv:1812.02056 (2018), Algorithm 2; the partial-pivoting protocol follows Golub and Van Loan, *Matrix Computations*, 4th edition, section 3.4 |
| Block classical Gram-Schmidt QR | Camarero, arXiv:1812.02056 (2018), Algorithm 3; the reorthogonalization bound is Giraud, Langou and Rozložník, *The loss of orthogonality in the Gram-Schmidt orthogonalization process*, Computers and Mathematics with Applications 50 (2005), 1069-1075 |

## Reference implementations consulted

| implementation | license | consulted for |
| --- | --- | --- |
| `LinearAlgebra` (Julia standard library) | MIT | the `lowrankupdate!`/`lowrankdowndate!` signatures this package extends |
| `QRupdate.jl` | MIT | API shape; benchmark comparison, maintaining `R` alone |
| `UpdatableQRFactorizations.jl` | MIT | API shape; benchmark comparison, maintaining a full `m x m` `Q` |
| `UpdatableCholeskyFactorizations.jl` | MIT | the capacity-with-active-block storage idea; benchmark comparison |

`qrupdate-ng` is GPL-3.0-or-later. Its source was not read. It serves only as a benchmark target
and as a reference list of routine names.

`QRupdatesFast.jl` links `qrupdate-ng` and so may appear only in a benchmark environment, never
as a dependency of this package.
