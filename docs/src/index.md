# UpdatableFactorizations.jl

UpdatableFactorizations.jl keeps a Cholesky or LU factorization current as the matrix it
factors is modified, instead of recomputing the factorization from scratch. It supports rank-1
update and downdate, and symmetric insertion, deletion and shifting of indices for Cholesky.

Building a factorization from scratch goes through `LinearAlgebra` (`cholesky`, `lu`): this
package does not offer a faster way to do that. Its value is in keeping an existing
factorization current under modification. Each update runs in `O(n^2)` operations, and does not
allocate once a factorization's scratch storage has been sized to the operation.

The Cholesky rank-1 update and downdate extend `LinearAlgebra.lowrankupdate!` and
`LinearAlgebra.lowrankdowndate!` with methods for `UpdatableCholesky`, rather than replacing the
existing methods those functions already provide for `Cholesky`.

QR support is planned but not yet implemented.

See [Getting started](@ref) to construct a factorization, [Updating and downdating](@ref) for a
worked example of each operation, and [Provenance](@ref) for the article each algorithm derives
from and the reference implementations consulted during development.
