@testitem "Cholesky symmetric deletion, every index" begin
    using LinearAlgebra
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        n = 7
        B = randn(T, n, n)
        A = Matrix(Hermitian(B * B' + n * I))
        for j in 1:n
            F = UpdatableCholesky(cholesky(Hermitian(A, uplo)))
            delete_column!(F, j)
            keep = [k for k in 1:n if k != j]
            L = Matrix(F)
            @test size(F) == (n - 1, n - 1)
            @test norm(L * L' - A[keep, keep]) / norm(A) < 1.0e-13
        end
    end
end

@testitem "Cholesky deletion rejects an out-of-range index" begin
    using LinearAlgebra
    B = randn(5, 5)
    F = UpdatableCholesky(cholesky(Symmetric(Matrix(Symmetric(B * B' + 5I)), :L)))
    @test_throws BoundsError delete_column!(F, 6)
    @test_throws BoundsError delete_column!(F, 0)
end

@testitem "Cholesky append at the end" begin
    using LinearAlgebra
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        n = 6
        B = randn(T, n + 1, n + 1)
        A = Matrix(Hermitian(B * B' + (n + 1) * I))
        F = UpdatableCholesky(cholesky(Hermitian(A[1:n, 1:n], uplo)))
        UpdatableFactorizations._append!(F, A[:, n + 1])
        L = Matrix(F)
        @test size(F) == (n + 1, n + 1)
        @test norm(L * L' - A) / norm(A) < 1.0e-13
    end
end

@testitem "Cholesky append grows past capacity" begin
    using LinearAlgebra
    n = 4
    B = randn(n + 1, n + 1)
    A = Matrix(Symmetric(B * B' + (n + 1) * I))
    F = UpdatableCholesky(cholesky(Symmetric(A[1:n, 1:n], :L)); capacity = n)
    UpdatableFactorizations._append!(F, A[:, n + 1])
    L = Matrix(F)
    @test norm(L * L' - A) / norm(A) < 1.0e-13
end

@testitem "Cholesky index shift, both directions" begin
    using LinearAlgebra
    n = 7
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        B = randn(T, n, n)
        A = Matrix(Hermitian(B * B' + n * I))
        for i in 1:n, j in 1:n
            F = UpdatableCholesky(cholesky(Hermitian(A, uplo)))
            shift_columns!(F, i, j)
            p = UpdatableFactorizations._cyclicperm(n, i, j)
            L = Matrix(F)
            @test norm(L * L' - A[p, p]) / norm(A) < 1.0e-12
        end
    end
end

@testitem "Cholesky symmetric insertion, every index" begin
    using LinearAlgebra
    n = 6
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        B = randn(T, n + 1, n + 1)
        A = Matrix(Hermitian(B * B' + (n + 1) * I))
        for j in 1:(n + 1)
            keep = [k for k in 1:(n + 1) if k != j]
            F = UpdatableCholesky(cholesky(Hermitian(A[keep, keep], uplo)))
            insert_column!(F, j, A[:, j])
            L = Matrix(F)
            @test size(F) == (n + 1, n + 1)
            @test norm(L * L' - A) / norm(A) < 1.0e-12
        end
    end
end
