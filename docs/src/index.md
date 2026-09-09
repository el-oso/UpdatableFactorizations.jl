# UpdatableFactorizations.jl

UpdatableFactorizations.jl keeps a Cholesky, LU or QR factorization current as the matrix it
factors is modified, instead of recomputing the factorization from scratch. It supports rank-1
update and downdate, symmetric insertion, deletion and shifting of indices for Cholesky, and
rank-1 update and column and row insertion, deletion and shifting for QR.

Building a factorization from scratch goes through `LinearAlgebra` (`cholesky`, `lu`, `qr`): this
package does not offer a faster way to do that. Its value is in keeping an existing
factorization current under modification. Each update runs in `O(n^2)` operations for Cholesky
and LU, or `O(mn)` for QR, and does not allocate once a factorization's scratch storage has been
sized to the operation.

The Cholesky rank-1 update and downdate extend `LinearAlgebra.lowrankupdate!` and
`LinearAlgebra.lowrankdowndate!` with methods for `UpdatableCholesky`, and the QR rank-1 update
extends `LinearAlgebra.lowrankupdate!` with a method for `UpdatableQR`, rather than replacing the
existing methods those functions already provide for `Cholesky` and `QR`.

See [Getting started](@ref) to construct a factorization, [Updating and downdating](@ref) for a
worked example of each operation, [Q representations](@ref) for how `UpdatableQR` stores its
orthonormal factor, and [Provenance](@ref) for the article each algorithm derives from and the
reference implementations consulted during development.
