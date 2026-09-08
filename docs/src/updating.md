# Updating and downdating

## Cholesky rank-1 update

`lowrankupdate!(F, v)` replaces the factorization of `A` with that of `A + v*v'`. It extends
`LinearAlgebra.lowrankupdate!`, so no new name needs to be imported once `LinearAlgebra` is in
scope.

```@example update
using LinearAlgebra, UpdatableFactorizations

A = [4.0 2.0; 2.0 3.0]
F = UpdatableCholesky(A)
v = [1.0, 0.5]
lowrankupdate!(F, v)
Matrix(F) ≈ A + v * v'
```

## Cholesky rank-1 downdate

`lowrankdowndate!(F, v)` replaces the factorization of `A` with that of `A - v*v'`. It throws
`PosDefException` when the result would not be positive definite, rather than returning a
factorization of a matrix that is not what was asked for:

```@example downdate
using LinearAlgebra, UpdatableFactorizations

A = [4.0 2.0; 2.0 3.0]
F = UpdatableCholesky(A)
try
    lowrankdowndate!(F, [10.0, 10.0])
catch e
    e
end
```

## Cholesky column insertion

`insert_column!(F, j, x)` adds index `j`, extending the factored matrix by both a row and a
column. `x` is the new row and column in the resulting `n+1` indexing, so `x[j]` is the new
diagonal entry:

```@example insert
using UpdatableFactorizations

A = [4.0 2.0; 2.0 3.0]
F = UpdatableCholesky(A)
insert_column!(F, 1, [10.0, 1.0, 2.0])
Matrix(F)
```

## Cholesky column deletion

`delete_column!(F, j)` removes index `j`, deleting both its row and column:

```@example insert
delete_column!(F, 1)
Matrix(F) ≈ A
```

## Cholesky column shift

`shift_columns!(F, i, j)` moves index `i` to position `j`, sliding the indices between them by
one, keeping the factored matrix symmetric:

```@example insert
shift_columns!(F, 1, 2)
Matrix(F) ≈ [3.0 2.0; 2.0 4.0]
```

## LU rank-1 update

`lowrankupdate!(F, u, v; rtol)` replaces the factorization of `A` with that of `A + u*v'`.
Dispatch on `F`'s pivoting picks the unpivoted or pivoted update automatically:

```@example lu1up
using LinearAlgebra, UpdatableFactorizations

B = [4.0 3.0; 6.0 3.0]
G = UpdatableLU(B)
u = [1.0, 0.0]
v = [0.0, 1.0]
lowrankupdate!(G, u, v)
Matrix(G) ≈ B + u * v'
```

### Failure and `rtol`

`ZeroPivotException` fires only on an exact zero pivot. A merely small pivot is accepted, and the
resulting error in the updated factors grows like `|d_k| / |d'_k|`, the ratio of a pivot before
and after the update. `rtol` widens the exact-zero test to a relative threshold, which is the
knob to use in a loop of updates rather than relying on the exact test alone:

```@example lu1upfail
using LinearAlgebra, UpdatableFactorizations

G = UpdatableLU([1.0 0.0; 0.0 1.0]; pivot = NoPivot())
try
    lowrankupdate!(G, [-1.0, 0.0], [1.0, 0.0])
catch e
    e
end
```

```@example lu1upfail
issuccess(G)
```

A throw leaves `G` invalid: `issuccess(G)` is `false`, `G.info` names the column at which the
update failed, and solving with or taking the determinant of `G` throws. An invalid
factorization can only be rebuilt, not repaired.
