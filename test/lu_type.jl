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

@testitem "UpdatableLU properties, reconstruction and determinant" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    for T in (Float64, ComplexF64)
        n = 5
        A = randn(T, n, n)
        F = UpdatableLU(lu(A))
        @test norm(Matrix(F) - A) / norm(A) < 1.0e-12
        @test F.U isa UpperTriangular
        @test norm(F.L * F.U - A[F.p, :]) / norm(A) < 1.0e-12
        # both properties are copies, so writing to one leaves the factorization alone
        L = F.L
        L[2, 1] += 1
        @test F.L != L
        @test issuccess(F)
        @test abs(det(F) - det(A)) / abs(det(A)) < 1.0e-10
        m, s = logabsdet(F)
        @test abs(exp(m) * s - det(A)) / abs(det(A)) < 1.0e-10
        @test propertynames(F) == (:L, :U, :p, :info)
        @test :Lf in propertynames(F, true)
    end
end

@testitem "UpdatableLU solves integer and matrix right-hand sides" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    A = randn(3, 3)
    F = UpdatableLU(lu(A))
    @test norm(A * (F \ [1, 2, 3]) - [1, 2, 3]) < 1.0e-10
    @test norm(A * (F \ Matrix(1.0I, 3, 3)) - I) < 1.0e-10
    @test norm(inv(F) - inv(A)) < 1.0e-10
end

@testitem "UpdatableLU at size zero" begin
    using LinearAlgebra
    F = UpdatableLU(lu(zeros(0, 0)))
    @test size(F) == (0, 0)
    @test size(F, 3) == 1
    @test det(F) == 1
    @test logdet(F) == 0
    @test_throws "dimension must be positive" size(F, 0)
end
