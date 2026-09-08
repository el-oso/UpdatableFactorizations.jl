@testitem "LU rank-1 update is type stable and allocates nothing" begin
    using LinearAlgebra, StrictModeTest
    n = 8
    G = UpdatableLU(lu(randn(n, n) + n * I))
    u = randn(n)
    v = randn(n)
    @test_typestable lowrankupdate!(G, u, v)
    @test_noalloc lowrankupdate!(G, u, v)
end

@testitem "factorization invariants hold after every operation" begin
    using LinearAlgebra, Test
    using UpdatableFactorizations: TypeContracts
    using .TypeContracts: behavior_passes
    n = 7
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))
    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    lowrankupdate!(F, randn(n))
    lowrankdowndate!(F, randn(n) ./ 16)
    delete_column!(F, 3)
    shift_columns!(F, 1, 4)
    # A large diagonal entry relative to the small off-diagonal keeps the inserted row and column
    # positive definite; a fully random insertion is not reliably so.
    x = randn(F.n + 1) ./ 10
    x[2] = n + 10.0
    insert_column!(F, 2, x)
    @test behavior_passes(UpdatableCholesky, [F])

    G = UpdatableLU(lu(randn(n, n) + n * I))
    lowrankupdate!(G, randn(n), randn(n))
    @test behavior_passes(UpdatableLU, [G])
end
