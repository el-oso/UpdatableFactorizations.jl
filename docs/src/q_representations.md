# Q representations

`UpdatableQR` does not store its orthonormal factor directly. It holds an `AbstractQRep`, which
answers five operations: its shape, writing itself densely into a matrix, producing an explicitly
stored equivalent, and applying a Givens rotation on the left or on the right. A thin factor
cannot subtype `LinearAlgebra.AbstractQ`, whose interface is the implicit square factor —
`size(qr(randn(6, 3)).Q)` is `(6, 6)` — so the seam is local to this package.

## `DenseQ`

The one representation. It stores the thin `m x n` factor explicitly in the leading block of an
over-allocated buffer, with one further column that the updating verbs build their augmentation
in. `materialize` returns it unchanged.

`F.Q` is a view of that block. It changes when a verb changes the factorization, and writing
through it changes the factorization.

The docstrings for [`AbstractQRep`](@ref UpdatableFactorizations.AbstractQRep),
[`DenseQ`](@ref UpdatableFactorizations.DenseQ) and
[`materialize`](@ref UpdatableFactorizations.materialize) are on the API page.
