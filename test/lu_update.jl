@testitem "LU rank-1 update, pivoted and not" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    for T in (Float64, ComplexF64), pivot in (true, false)
        n = 7
        A = pivot ? randn(T, n, n) : randn(T, n, n) + n * I
        u = randn(T, n)
        v = randn(T, n)
        F = UpdatableLU(pivot ? lu(A) : lu(A, NoPivot()))
        if pivot
            @test F.p != 1:n
        end
        lowrankupdate!(F, u, v)
        @test norm(F.L * F.U - (A + u * v')[F.p, :]) / norm(A) < 1.0e-11
    end
end

@testitem "LU rank-1 update does not consume its vectors" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 6
    A = randn(n, n)
    u = randn(n)
    v = randn(n)
    ucopy, vcopy = copy(u), copy(v)
    F = UpdatableLU(lu(A))
    lowrankupdate!(F, u, v)
    @test u == ucopy
    @test v == vcopy
end

@testitem "LU rank-1 update rejects mismatched vectors" begin
    using LinearAlgebra
    F = UpdatableLU(lu(randn(6, 6) + 6I))
    @test_throws "u has length 5, factorization is 6" lowrankupdate!(F, randn(5), randn(6))
    @test_throws "v has length 7, factorization is 6" lowrankupdate!(F, randn(6), randn(7))
end
