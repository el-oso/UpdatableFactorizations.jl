# Runs in a fresh Julia process, on a temporary project (built here, from `ARGS[1]`, the package
# root) whose LocalPreferences.toml sets `StrictMode.checks_enabled = false` — the configuration
# the package ships in. Measures every updating verb's allocation with `@allocated` and prints
# one "label\tbytes" line per measurement, plus a "checks_enabled\t<bool>" line recording whether
# the preference actually took effect. `test/strict.jl`'s "updating verbs allocate nothing with
# checks disabled" testitem launches this as a subprocess and asserts on its output; nothing here
# calls `@test` itself, since a `Test` failure inside a discarded worker process would be
# invisible to the parent.
#
# `StrictMode` must be a direct dependency of this project, not merely a transitive one through
# `UpdatableFactorizations`: Preferences.jl only reads a `LocalPreferences.toml` entry for
# packages an environment depends on directly.

using Pkg
Pkg.develop(path = ARGS[1])
Pkg.add("StrictMode")

using UpdatableFactorizations, LinearAlgebra, Random
using UpdatableFactorizations: StrictMode

println("checks_enabled\t", StrictMode.checks_enabled())

report(label, bytes) = println(label, "\t", bytes)

function measure_qr(T)
    Random.seed!(20260908)
    m, n = 80, 50
    mk() = UpdatableQR(randn(T, m, n))
    u, v = randn(T, m), randn(T, n)
    x, row = randn(T, m), randn(T, n)
    for F in (mk(), mk(), mk(), mk(), mk(), mk())   # compile every kernel before measuring
        lowrankupdate!(F, u, v)
        insert_column!(F, 2, x)
        delete_column!(F, 3)
        shift_columns!(F, 1, 4)
        insert_row!(F, 2, row)
        delete_row!(F, 5)
    end
    A, B, C, D, E, G = mk(), mk(), mk(), mk(), mk(), mk()
    report("QR lowrankupdate! $T", @allocated lowrankupdate!(A, u, v))
    report("QR insert_column! $T", @allocated insert_column!(B, 2, x))
    report("QR delete_column! $T", @allocated delete_column!(C, 3))
    report("QR shift_columns! $T", @allocated shift_columns!(D, 1, 4))
    report("QR insert_row! $T", @allocated insert_row!(E, 2, row))
    report("QR delete_row! $T", @allocated delete_row!(G, 5))
    return nothing
end

function measure_cholesky(T)
    Random.seed!(20260908)
    n = 40
    Bmat = randn(T, n, n)
    A = Matrix(Hermitian(Bmat * Bmat' + n * I))
    mk() = UpdatableCholesky(cholesky(Hermitian(A, :L)))
    v = randn(T, n) ./ 4
    for _ in 1:6                            # compile every kernel before measuring
        lowrankupdate!(mk(), v)
        lowrankdowndate!(mk(), v)
    end
    C1, C2 = mk(), mk()
    report("Cholesky lowrankupdate! $T", @allocated lowrankupdate!(C1, v))
    report("Cholesky lowrankdowndate! $T", @allocated lowrankdowndate!(C2, v))

    n1 = n + 1
    B2 = randn(T, n1, n1)
    A2 = Matrix(Hermitian(B2 * B2' + n1 * I))
    j = 3
    keep = [k for k in 1:n1 if k != j]
    mk_full() = UpdatableCholesky(cholesky(Hermitian(A2, :L)))               # size n+1
    mk_reduced() = UpdatableCholesky(cholesky(Hermitian(A2[keep, keep], :L))) # size n
    col = A2[:, j]
    for _ in 1:6                            # compile every kernel before measuring
        delete_column!(mk_full(), j)
        shift_columns!(mk_full(), 1, n)
        insert_column!(mk_reduced(), j, col)
    end
    D1, D2, D3 = mk_full(), mk_full(), mk_reduced()
    report("Cholesky delete_column! $T", @allocated delete_column!(D1, j))
    report("Cholesky shift_columns! $T", @allocated shift_columns!(D2, 1, n))
    report("Cholesky insert_column! $T", @allocated insert_column!(D3, j, col))
    return nothing
end

function measure_lu(T)
    Random.seed!(20260908)
    n = 40
    mk() = UpdatableLU(lu(randn(T, n, n) + n * I))
    u, v = randn(T, n), randn(T, n)
    for _ in 1:6                            # compile every kernel before measuring
        lowrankupdate!(mk(), u, v)
    end
    G = mk()
    report("LU lowrankupdate! $T", @allocated lowrankupdate!(G, u, v))
    return nothing
end

for T in (Float64, ComplexF64)
    measure_qr(T)
    measure_cholesky(T)
    measure_lu(T)
end
