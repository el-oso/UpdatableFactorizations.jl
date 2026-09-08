@testitem "Cholesky rank-1 update" begin
    using LinearAlgebra
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        B = randn(T, 7, 7)
        A = Matrix(Hermitian(B * B' + 7I))
        v = randn(T, 7)
        F = UpdatableCholesky(cholesky(Hermitian(A, uplo)))
        lowrankupdate!(F, v)
        L = Matrix(F)
        @test norm(L * L' - (A + v * v')) / norm(A) < 1.0e-13
    end
end

@testitem "Cholesky rank-1 update does not consume its vector" begin
    using LinearAlgebra
    B = randn(6, 6)
    A = Matrix(Symmetric(B * B' + 6I))
    v = randn(6)
    vcopy = copy(v)
    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    lowrankupdate!(F, v)
    @test v == vcopy
end

@testitem "Cholesky rank-1 update rejects a mismatched vector" begin
    using LinearAlgebra
    B = randn(6, 6)
    F = UpdatableCholesky(cholesky(Symmetric(Matrix(Symmetric(B * B' + 6I)), :L)))
    @test_throws "has length 5, factorization is 6" lowrankupdate!(F, randn(5))
end

@testitem "Cholesky rank-1 downdate" begin
    using LinearAlgebra
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        B = randn(T, 7, 7)
        A = Matrix(Hermitian(B * B' + 7I))
        v = randn(T, 7) ./ 8
        F = UpdatableCholesky(cholesky(Hermitian(A, uplo)))
        lowrankdowndate!(F, v)
        L = Matrix(F)
        @test norm(L * L' - (A - v * v')) / norm(A) < 1.0e-13
    end
end

@testitem "Cholesky downdate past positive definiteness throws" begin
    using LinearAlgebra
    B = randn(6, 6)
    A = Matrix(Symmetric(B * B' + 6I))
    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    # A - v*v' with v far larger than any direction of A cannot be positive definite.
    @test_throws PosDefException lowrankdowndate!(F, randn(6) .* 1000)
end
