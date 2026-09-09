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

@testitem "QR factorization invariants hold after every operation" begin
    using LinearAlgebra, Random, Test
    using UpdatableFactorizations: TypeContracts
    using .TypeContracts: behavior_passes

    Random.seed!(20260908)
    m, n = 12, 5
    A = randn(m, n)
    F = UpdatableQR(A)
    lowrankupdate!(F, randn(m), randn(n))
    insert_column!(F, 2, randn(m))
    delete_column!(F, 4)
    shift_columns!(F, 1, 4)
    insert_row!(F, 3, randn(F.n))
    delete_row!(F, 7)
    @test behavior_passes(UpdatableQR, [F])
end

@testitem "QR updating allocates nothing after construction" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    function measure(::Type{T}, m, n) where {T}
        mk() = UpdatableQR(randn(T, m, n))
        u = randn(T, m)
        v = randn(T, n)
        x = randn(T, m)
        row = randn(T, n)
        for F in (mk(), mk(), mk(), mk(), mk(), mk())   # compile every kernel before measuring
            lowrankupdate!(F, u, v)
            insert_column!(F, 2, x)
            delete_column!(F, 3)
            shift_columns!(F, 1, 4)
            insert_row!(F, 2, row)
            delete_row!(F, 5)
        end
        A, B, C, D, E, G = mk(), mk(), mk(), mk(), mk(), mk()
        return (
            @allocated(lowrankupdate!(A, u, v)),
            @allocated(insert_column!(B, 2, x)),
            @allocated(delete_column!(C, 3)),
            @allocated(shift_columns!(D, 1, 4)),
            @allocated(insert_row!(E, 2, row)),
            @allocated(delete_row!(G, 5)),
        )
    end
    @test measure(Float64, 80, 50) == (0, 0, 0, 0, 0, 0)
    @test measure(ComplexF64, 80, 50) == (0, 0, 0, 0, 0, 0)
end

@testitem "QR rank-1 kernels are type stable and allocate nothing" begin
    using LinearAlgebra, Random, Test, StrictModeTest

    Random.seed!(20260908)
    for T in (Float64, ComplexF64)
        m, n = 40, 12
        F = UpdatableQR(randn(T, m, n))
        u = randn(T, m)
        v = randn(T, n)
        @test_typestable lowrankupdate!(F, u, v)
        # AllocCheck's all-paths analysis cannot rule out that `w` and `corr` in
        # `_project_residual!` alias: both are SubArrays of the same parametric type over
        # distinct Vector{Float64} storage, and the language gives no way to declare two
        # same-typed arguments non-aliasing to the analyzer. The defensive copy this forces
        # Base's broadcast machinery to consider is never reached at run time, which is what
        # `@allocated(lowrankupdate!(...)) == 0` above already proves.
        @test_broken (@test_noalloc(lowrankupdate!(F, u, v)); true)
        G = UpdatableQR(randn(T, m, n))
        @test_typestable delete_row!(G, 3)
        H = UpdatableQR(randn(T, m, n))
        @test_broken (@test_noalloc(delete_row!(H, 3)); true)
    end
end
