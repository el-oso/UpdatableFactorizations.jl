@testitem "LU rank-1 update is type stable and allocates nothing" begin
    using LinearAlgebra, StrictModeTest, Random
    Random.seed!(20260908)
    n = 8
    G = UpdatableLU(lu(randn(n, n) + n * I))
    u = randn(n)
    v = randn(n)
    @test_typestable lowrankupdate!(G, u, v)
    @test_noalloc lowrankupdate!(G, u, v)
end

@testitem "factorization invariants hold after every operation" begin
    using LinearAlgebra, Test, Random
    using UpdatableFactorizations: TypeContracts
    using .TypeContracts: behavior_passes
    Random.seed!(20260908)
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

@testitem "Cholesky resizing allocates nothing after construction" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    function measure(n)
        function mk()
            B = randn(n, n)
            return UpdatableCholesky(cholesky(Symmetric(B * B' + n * I)))
        end
        append = zeros(n + 1)
        append[n + 1] = 100.0
        insert = zeros(n + 1)
        insert[2] = 100.0
        for F in (mk(), mk(), mk(), mk())      # compile every kernel before measuring
            delete_column!(F, 3)
        end
        A, B, C, D = mk(), mk(), mk(), mk()
        shift_columns!(A, 1, n)
        UpdatableFactorizations._append!(B, append)
        insert_column!(C, 2, insert)
        E, G, H, K = mk(), mk(), mk(), mk()
        return (
            @allocated(delete_column!(E, 3)),
            @allocated(shift_columns!(G, 1, n)),
            @allocated(UpdatableFactorizations._append!(H, append)),
            @allocated(insert_column!(K, 2, insert)),
        )
    end
    @test measure(50) == (0, 0, 0, 0)
end
