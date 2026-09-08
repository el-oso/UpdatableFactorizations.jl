@testitem "UpdatableCholesky reconstructs its matrix" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        B = randn(T, 6, 6)
        A = Matrix(Hermitian(B * B' + 6I))
        F = UpdatableCholesky(cholesky(Hermitian(A, uplo)))
        L = F.L
        @test norm(L * L' - A) / norm(A) < 1.0e-13
        @test size(F) == (6, 6)
    end
end

@testitem "UpdatableCholesky ignores the unstored triangle" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    B = randn(6, 6)
    A = Matrix(Symmetric(B * B' + 6I))
    C = cholesky(Symmetric(A, :L))
    # cholesky leaves the original matrix in the strict upper triangle; it must not leak in.
    @test any(!iszero, [C.factors[i, j] for i in 1:6 for j in (i + 1):6])
    F = UpdatableCholesky(C)
    @test all(iszero, [F.factors[i, j] for i in 1:6 for j in (i + 1):6])
end

@testitem "UpdatableCholesky solves" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    B = randn(6, 6)
    A = Matrix(Symmetric(B * B' + 6I))
    b = randn(6)
    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    @test norm(A * (F \ b) - b) / norm(b) < 1.0e-12
end

@testitem "UpdatableCholesky reconstructs the factored matrix" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        B = randn(T, 5, 5)
        A = Matrix(Hermitian(B * B' + 5I))
        F = UpdatableCholesky(cholesky(Hermitian(A, uplo)))
        @test Matrix(F) isa Matrix{T}
        @test norm(Matrix(F) - A) / norm(A) < 1.0e-13
        @test F.U == F.L'
        @test hasproperty(F, :L) && hasproperty(F, :U)
        @test :factors in propertynames(F, true)
        @test issuccess(F)
    end
end

@testitem "UpdatableCholesky solves integer and matrix right-hand sides" begin
    using LinearAlgebra
    A = Float64[4 1 0; 1 3 1; 0 1 2]
    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    @test norm(A * (F \ [1, 2, 3]) - [1, 2, 3]) < 1.0e-12
    @test norm(A * (F \ Matrix(1.0I, 3, 3)) - I) < 1.0e-12
end

@testitem "UpdatableCholesky at size zero" begin
    using LinearAlgebra
    F = UpdatableCholesky(cholesky(Symmetric(zeros(0, 0))))
    @test size(F) == (0, 0)
    @test size(F, 3) == 1
    @test logdet(F) == 0
    @test det(F) == 1
    @test_throws "dimension must be positive" size(F, 0)
end
