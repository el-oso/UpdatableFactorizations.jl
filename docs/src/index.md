# UpdatableFactorizations.jl

UpdatableFactorizations.jl keeps a Cholesky, LU or QR factorization current as the matrix it
factors is modified, instead of recomputing the factorization from scratch. It supports rank-1
update and downdate, symmetric insertion, deletion and shifting of indices for Cholesky, and
rank-1 update and column and row insertion, deletion and shifting for QR.

Building a factorization from scratch goes through `LinearAlgebra` (`cholesky`, `lu`, `qr`): this
package does not offer a faster way to do that. Its value is in keeping an existing
factorization current under modification. Each update runs in `O(n^2)` operations for Cholesky
and LU, or `O(mn)` for QR. The underlying kernels are allocation-free once a factorization's
scratch storage has been sized to the operation; the guarded rank-1 entry points reach that same
allocation-free behavior only when `StrictMode.checks_enabled()` is false — see
[Updating and downdating](@ref) for the measured byte counts.

The Cholesky rank-1 update and downdate extend `LinearAlgebra.lowrankupdate!` and
`LinearAlgebra.lowrankdowndate!` with methods for `UpdatableCholesky`, and the QR rank-1 update
extends `LinearAlgebra.lowrankupdate!` with a method for `UpdatableQR`, rather than replacing the
existing methods those functions already provide for `Cholesky` and `QR`.

The package also builds a factorization from scratch with three algorithms from Camarero,
arXiv:1812.02056: a blocked Crout Cholesky, a blocked Crout LU, and a block classical
Gram-Schmidt QR. Most of these are slower than the standard library's LAPACK calls; the
exception is the QR routine without reorthogonalization, which is faster than forming a
factorization through `LinearAlgebra.qr`. See [Construction](@ref) for the measured ratios and
accuracy.

See [Getting started](@ref) to construct a factorization, [Construction](@ref) for the
Camarero-derived construction routines, [Updating and downdating](@ref) for a worked example of
each update operation, [Q representations](@ref) for how `UpdatableQR` stores its orthonormal
factor, and [Provenance](@ref) for the article each algorithm derives from and the reference
implementations consulted during development.
