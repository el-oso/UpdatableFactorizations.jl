# Getting started

```julia
using Pkg
Pkg.add("UpdatableFactorizations")
```

## Cholesky

`UpdatableCholesky` factors a Hermitian positive definite matrix. It can be built from a plain
matrix or from an existing `LinearAlgebra.Cholesky`; both input orientations, `uplo = :L` and
`uplo = :U`, give identical results because the package always stores the lower factor.

```@example cholesky
using LinearAlgebra, UpdatableFactorizations

A = [4.0 2.0; 2.0 3.0]
F = UpdatableCholesky(A)
F.L
```

`F.U` is the adjoint of `F.L`, and `Matrix(F)` reconstructs `A`:

```@example cholesky
F.U
```

```@example cholesky
Matrix(F) ≈ A
```

`F` supports the standard `Factorization` interface — `\`, `ldiv!`, `size`, `det`, `logdet`:

```@example cholesky
b = [1.0, 2.0]
F \ b
```

## LU

`UpdatableLU` factors a general square matrix, with partial pivoting by default. It is stored as
`P*A = L*Diagonal(d)*U`, but `F.L` and `F.U` return the factors in the same form
`LinearAlgebra.lu` gives them, so the two are interchangeable for reading off the factorization.

```@example lu
using LinearAlgebra, UpdatableFactorizations

B = [4.0 3.0; 6.0 3.0]
G = UpdatableLU(B)
G.L
```

```@example lu
G.U
```

```@example lu
Matrix(G) ≈ B
```

`G.info` and `issuccess(G)` report whether the factorization is still valid; see
[Updating and downdating](@ref) for what invalidates it.

## Capacity

`UpdatableCholesky` can grow: `insert_column!` extends the factored matrix by one index. The
`capacity` keyword sets how large the factorization can grow before its storage is reallocated:

```@example cholesky
F2 = UpdatableCholesky(A; capacity = 10)
size(F2)
```

`UpdatableLU` has no resizing operation and so takes no `capacity` argument.
