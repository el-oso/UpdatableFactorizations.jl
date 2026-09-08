@testitem "UpdatableLU reconstructs its matrix" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    for T in (Float64, ComplexF64), pivot in (true, false)
        n = 6
        A = if pivot
            randn(T, n, n)
        else
            randn(T, n, n) + n * I
        end
        G = pivot ? lu(A) : lu(A, NoPivot())
        F = UpdatableLU(G)
        @test norm(F.L * F.U - A[F.p, :]) / norm(A) < 1.0e-12
        @test size(F) == (n, n)
        if pivot
            @test F.p != 1:n
        end
    end
end

@testitem "UpdatableLU solves" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 6
    A = randn(n, n)
    b = randn(n)
    F = UpdatableLU(lu(A))
    @test norm(A * (F \ b) - b) / norm(b) < 1.0e-11
    @test F.p != 1:n
end
