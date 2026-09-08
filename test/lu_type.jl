@testitem "UpdatableLU reconstructs its matrix" begin
    using LinearAlgebra
    for T in (Float64, ComplexF64), pivot in (true, false)
        n = 6
        A = randn(T, n, n) + n * I
        G = pivot ? lu(A) : lu(A, NoPivot())
        F = UpdatableLU(G)
        @test norm(F.L * F.U - A[F.p, :]) / norm(A) < 1.0e-12
        @test size(F) == (n, n)
    end
end

@testitem "UpdatableLU solves" begin
    using LinearAlgebra
    n = 6
    A = randn(n, n) + n * I
    b = randn(n)
    F = UpdatableLU(lu(A))
    @test norm(A * (F \ b) - b) / norm(b) < 1.0e-11
end
