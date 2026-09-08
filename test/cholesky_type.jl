@testitem "UpdatableCholesky reconstructs its matrix" begin
    using LinearAlgebra
    for uplo in (:L, :U), T in (Float64, ComplexF64)
        B = randn(T, 6, 6)
        A = Matrix(Hermitian(B * B' + 6I))
        F = UpdatableCholesky(cholesky(Hermitian(A, uplo)))
        L = Matrix(F)
        @test norm(L * L' - A) / norm(A) < 1.0e-13
        @test size(F) == (6, 6)
    end
end

@testitem "UpdatableCholesky ignores the unstored triangle" begin
    using LinearAlgebra
    B = randn(6, 6)
    A = Matrix(Symmetric(B * B' + 6I))
    C = cholesky(Symmetric(A, :L))
    # cholesky leaves the original matrix in the strict upper triangle; it must not leak in.
    @test any(!iszero, [C.factors[i, j] for i in 1:6 for j in (i + 1):6])
    F = UpdatableCholesky(C)
    @test all(iszero, [F.factors[i, j] for i in 1:6 for j in (i + 1):6])
end

@testitem "UpdatableCholesky solves" begin
    using LinearAlgebra
    B = randn(6, 6)
    A = Matrix(Symmetric(B * B' + 6I))
    b = randn(6)
    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    @test norm(A * (F \ b) - b) / norm(b) < 1.0e-12
end
