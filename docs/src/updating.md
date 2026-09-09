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

## QR construction

`UpdatableQR` factors a thin, `m >= n` matrix, and supports rank-1 update and insertion,
deletion and shifting of columns and rows. `qr_householder(A)` is the recommended way to build
one: it factors through `LinearAlgebra.qr`, so orthogonality of `Q` is at machine precision
whatever the condition number of `A`, which no Gram-Schmidt variant gives.

```@example qrconstruct
using LinearAlgebra, UpdatableFactorizations

A = [4.0 2.0; 2.0 3.0; 1.0 1.0; 0.0 2.0]
F = qr_householder(A)
Matrix(F) ≈ A
```

```@example qrconstruct
norm(F.Q'F.Q - I) < 1.0e-14
```

Unlike `UpdatableLU`'s `F.L` and `F.U`, which are reassembled on access, `F.Q` and `F.R` are
views of live storage: a later verb changes what an earlier one returned, and writing through
them changes the factorization.

`capacity` is the pair `(mcap, ncap)`, the largest shape the factorization reaches before its
storage is reallocated:

```@example qrconstruct
UpdatableFactorizations.capacity(F)
```

Growing `ncap` appends columns to a column-major buffer; growing `mcap` re-strides every
existing column, so row growth is the expensive direction. `insert_row!` is the only verb that
grows it, so a caller inserting rows in a loop should pre-size `capacity` rather than let it
double repeatedly.

Every verb either completes or throws with the factorization left exactly as it was, so
`issuccess(F)` is always `true` for an `UpdatableQR` — there is no invalid state to recover
from, unlike `UpdatableLU`. Three of the six verbs reorthogonalize with an unconditional second
pass against `Q`: this costs a second `O(mn)` pair of matrix-vector products in
`lowrankupdate!`, `insert_column!` and `delete_row!`, and it is what keeps `norm(F.Q'F.Q - I)`
at the level of rounding across the whole range from a well-conditioned update to one at the
edge of the verb's own refusal. The package does not build a factorization faster than
`LinearAlgebra.qr`, and this extra pass widens that gap further for the three verbs that pay it.

## QR rank-1 update

`lowrankupdate!(F, u, v; rtol)` replaces the factorization of `A` with that of `A + u*v'`.
Neither `u` nor `v` is modified:

```@example qrupdate
using LinearAlgebra, UpdatableFactorizations

A = [4.0 2.0; 2.0 3.0; 1.0 1.0; 0.0 2.0]
F = qr_householder(A)
u = [1.0, 0.5, -0.5, 0.25]
v = [0.5, -1.0]
lowrankupdate!(F, u, v)
Matrix(F) ≈ A + u * v'
```

### `rtol`

`lowrankupdate!`, `insert_column!` and `delete_row!` all carry an `rtol` keyword, defaulting to
`sqrt(eps(real(T)))`. Each compares a measured quantity against `rtol` and throws
`ArgumentError`, naming both the quantity and the row or column, when the update would leave the
factorization exact but numerically rank deficient: a later solve through the collapsed
direction would divide by rounding noise rather than by a real pivot. Passing `rtol = 0`
disables the check for `lowrankupdate!` and `insert_column!`, admitting the update or insertion
outright, including an exactly singular one.

For `lowrankupdate!`, the measured quantity is `abs(R[j,j]) / norm(R[1:j,j])` for the first
column `j` at which it collapses — `1` when column `j` carries no contribution from the columns
before it, falling toward `0` as `u*v'` drives it into their span:

```@example qrupdate2
using LinearAlgebra, UpdatableFactorizations

Q = [1.0 0.0; 0.0 1.0; 0.0 0.0]
R = [3.0 1.0; 0.0 5.0]
F = UpdatableQR(Q, R)
u = [0.0, -5.0 * (1 - 1.0e-10), 0.0]
v = [0.0, 1.0]
try
    lowrankupdate!(F, u, v)
catch e
    e
end
```

The throw left `F` untouched:

```@example qrupdate2
issuccess(F) && Matrix(F) ≈ Q * R
```

`rtol = 0` admits the same update, producing the numerically singular result the default
refuses:

```@example qrupdate2
G = UpdatableQR(Q, R)
lowrankupdate!(G, u, v; rtol = 0.0)
Matrix(G) ≈ [3.0 1.0; 0.0 5.0e-10; 0.0 0.0]
```

### Allocation

`lowrankupdate!` is the one QR verb guarded by `@strict`, StrictMode's type-stability and
allocation-freedom check — the same guard covers Cholesky's `lowrankupdate!` and
`lowrankdowndate!` and LU's `lowrankupdate!`. With `StrictMode.checks_enabled()` true — the
setting this package's own test suite develops under — the guard's own reflection is real
compiled code in the guarded body, and it allocates about 4240 bytes of StrictMode's own
bookkeeping per call in steady state, independent of `F`'s size. The first call to a given method
signature is far larger, because the scan is cached per signature rather than repeated: about
31.9 MB for Cholesky and 4.14 MB for LU. With `checks_enabled` false — the configuration a
shipped build sets — that guard expands to the bare call, and the measured allocation is exactly
zero on every call, including the first, for Cholesky, LU and QR. Both figures describe the
guard, not the underlying kernel, which is allocation-free either way. The other five QR verbs
carry no `@strict` guard and measure zero bytes regardless of this setting.

## QR column insertion

`insert_column!(F, j, x; rtol)` inserts `x` as column `j`, sliding the existing columns at or
past `j` one place to the right. `x` is not modified:

```@example qrcol
using LinearAlgebra, UpdatableFactorizations

A = [4.0 2.0; 2.0 3.0; 1.0 1.0; 0.0 2.0]
F = qr_householder(A)
insert_column!(F, 1, [10.0, 1.0, 2.0, 3.0])
Matrix(F) ≈ [10.0 4.0 2.0; 1.0 2.0 3.0; 2.0 1.0 1.0; 3.0 0.0 2.0]
```

`DimensionMismatch` is thrown when the factorization is square, because inserting a column would
leave `m < n + 1`, and the type requires `m >= n`:

```@example qrcolsquare
using LinearAlgebra, UpdatableFactorizations

G = qr_householder([1.0 2.0; 3.0 4.0])
try
    insert_column!(G, 1, [1.0, 1.0])
catch e
    e
end
```

```@example qrcolsquare
issuccess(G)
```

For `insert_column!`, `rtol`'s measured quantity is `rho / norm(x)`, where `rho` is the norm of
the part of `x` orthogonal to the existing columns. A column that lies in the range of the
existing ones — an exact linear combination of them — leaves `rho` at the level of rounding
rather than at zero, so the default `rtol` refuses it:

```@example qrcol2
using LinearAlgebra, UpdatableFactorizations

A = [4.0 2.0; 2.0 3.0; 1.0 1.0; 0.0 2.0]
F = qr_householder(A)
x = F.Q[:, 1]
try
    insert_column!(F, 1, x)
catch e
    e
end
```

```@example qrcol2
size(F) == (4, 2)   # the throw left the factorization as it was
```

## QR column deletion

`delete_column!(F, j)` removes column `j`, sliding the columns after it one place to the left:

```@example qrcol
delete_column!(F, 1)
Matrix(F) ≈ A
```

## QR column shift

`shift_columns!(F, i, j)` moves column `i` to position `j`, sliding the columns between them by
one:

```@example qrcol
shift_columns!(F, 1, 2)
Matrix(F) ≈ [2.0 4.0; 3.0 2.0; 1.0 1.0; 2.0 0.0]
```

## QR row insertion

`insert_row!(F, i, x)` inserts `x` as row `i`, sliding the rows at or past `i` one place down.
`x` is the new row, not its adjoint, and is not modified. This is the verb that grows the row
capacity, which re-strides every column of the stored factor, so a caller inserting rows in a
loop should pre-size `capacity`:

```@example qrrow
using LinearAlgebra, UpdatableFactorizations

A = [4.0 2.0; 2.0 3.0; 1.0 1.0; 0.0 2.0]
F = qr_householder(A)
insert_row!(F, 2, [5.0, 6.0])
Matrix(F) ≈ [4.0 2.0; 5.0 6.0; 2.0 3.0; 1.0 1.0; 0.0 2.0]
```

## QR row deletion

`delete_row!(F, i; rtol)` removes row `i`:

```@example qrrow
delete_row!(F, 2)
Matrix(F) ≈ A
```

`DimensionMismatch` is thrown when the factorization is square, because deleting a row would
leave `m - 1 < n`:

```@example qrrowsquare
using LinearAlgebra, UpdatableFactorizations

H = qr_householder([1.0 2.0; 3.0 4.0])
try
    delete_row!(H, 1)
catch e
    e
end
```

For `delete_row!`, `rtol`'s measured quantity is `gamma`, the norm of the part of `e_i`
orthogonal to the range of `Q`: it lies in `[0, 1]` and agrees with the smallest singular value
of the remaining rows to several digits. A row at leverage one — whose removal would drop the
numerical rank of what remains — leaves `gamma` computed at the level of rounding, visible in
the error message below, rather than at exactly zero, so `rtol = 0` would make this refusal
unreachable for exactly this case, unlike `lowrankupdate!` and `insert_column!` where it
reliably disables the check:

```@example qrrowlev
using LinearAlgebra, UpdatableFactorizations

# Row 3 is twice row 2, so rows 2 and 3 together have rank 1: row 1 alone carries the
# factorization's second dimension, and removing it drops the numerical rank.
A = [1.0 0.0; 1.0 1.0; 2.0 2.0]
G = UpdatableQR(A)
try
    delete_row!(G, 1)
catch e
    e
end
```

```@example qrrowlev
issuccess(G)
```

Row 3, the dependent one, carries no leverage of its own and deletes cleanly:

```@example qrrowlev
delete_row!(G, 3)
size(G)
```
