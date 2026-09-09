@testitem "default_rankk! computes a symmetric rank-k update" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    for T in (Float64, Float32, ComplexF64, BigFloat)
        A = randn(T, 9, 4)
        C0 = Matrix(Hermitian(randn(T, 9, 9)))
        want = -A * A' + C0
        C = copy(C0)
        UpdatableFactorizations.default_rankk!(C, A, -one(T), one(T); uplo = 'L')
        # The BLAS path writes the lower triangle only, so only that triangle is compared.
        @test norm(tril(C) - tril(want)) / norm(want) < 100 * eps(real(T))
    end
end

@testitem "default_rankk! leaves the untouched triangle alone on BLAS element types" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    A = randn(9, 4)
    C = zeros(9, 9)
    C[1, 9] = 17.0
    UpdatableFactorizations.default_rankk!(C, A, -1.0, 1.0; uplo = 'L')
    @test C[1, 9] == 17.0
end

@testitem "default_rankk! writes both triangles for non-BLAS element types" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    A = randn(BigFloat, 9, 4)
    C0 = Matrix(Hermitian(randn(BigFloat, 9, 9)))
    want = -A * A' + C0
    C = copy(C0)
    UpdatableFactorizations.default_rankk!(C, A, -one(BigFloat), one(BigFloat); uplo = 'L')
    # No symmetric rank-k kernel exists for BigFloat, so the fallback forms the full product:
    # the upper triangle equals `want` too, not just the lower triangle named by `uplo`.
    @test norm(triu(C) - triu(want)) / norm(want) < 100 * eps(BigFloat)
end

@testitem "default_rankk! BLAS and generic fallback agree on the same values" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)

    # An AbstractMatrix that is not a StridedMatrix, so dispatch cannot reach the syrk!/herk!
    # methods regardless of element type: this forces the generic `mul!`-based fallback even
    # though the element type (Float64) would otherwise use the BLAS path.
    struct NoStride{T} <: AbstractMatrix{T}
        data::Matrix{T}
    end
    Base.size(A::NoStride) = size(A.data)
    Base.getindex(A::NoStride, i::Int, j::Int) = A.data[i, j]

    A = randn(9, 4)
    C0 = Matrix(Hermitian(randn(9, 9)))

    C_blas = copy(C0)
    UpdatableFactorizations.default_rankk!(C_blas, A, -1.0, 1.0; uplo = 'L')

    C_generic = copy(C0)
    UpdatableFactorizations.default_rankk!(C_generic, NoStride(A), -1.0, 1.0; uplo = 'L')

    @test norm(tril(C_blas) - tril(C_generic)) / norm(tril(C_blas)) < 100 * eps()
end

@testitem "cholesky_crout factors" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 9
    for uplo in (:L, :U), T in (Float64, ComplexF64), s in (1, 4, 64)
        B = randn(T, n, n)
        A = Matrix(Hermitian(B * B' + n * I))
        # The triangle `uplo` does not name is overwritten with garbage, so an implementation
        # that reads the wrong triangle, or ignores the keyword, cannot pass both halves of the
        # loop. `A` is the Hermitian completion of the triangle that is read.
        G = copy(A)
        for j in 1:n, i in 1:(j - 1)
            uplo === :L ? (G[i, j] = 1.0e6 * (i + j)) : (G[j, i] = 1.0e6 * (i + j))
        end
        F = cholesky_crout(G; s, uplo)
        L = Matrix(F.L)
        @test norm(L * L' - A) / norm(A) < 1.0e-13
        @test all(x -> abs(imag(x)) < 1.0e-12 && real(x) > 0, diag(L))
        @test size(F) == (n, n)
        @test UpdatableFactorizations.capacity(F) == 2n
        @test norm(A * (F \ ones(T, n)) - ones(T, n)) < 1.0e-11
    end
end

@testitem "cholesky_crout honors capacity and keeps updating available" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 6
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))
    F = cholesky_crout(A; s = 2, capacity = 20)
    @test UpdatableFactorizations.capacity(F) == 20
    v = randn(n)
    lowrankupdate!(F, v)
    @test norm(Matrix(F) - (A + v * v')) / norm(A) < 1.0e-13
end

@testitem "cholesky_crout throws on a matrix that is not positive definite" begin
    using LinearAlgebra
    A = Matrix(Symmetric([1.0 2.0 0.0; 2.0 1.0 0.0; 0.0 0.0 1.0]))
    # The second pivot is the first non-positive one, and the exception names it.
    @test_throws PosDefException(2) cholesky_crout(A; s = 2)
end

@testitem "cholesky_crout rejects a block size below one" begin
    using LinearAlgebra
    A = Matrix(1.0I, 4, 4)
    @test_throws "block size s must be at least 1, got 0" cholesky_crout(A; s = 0)
end

@testitem "cholesky_crout rejects an offset matrix" begin
    using LinearAlgebra, OffsetArrays
    A = OffsetMatrix(Matrix(1.0I, 4, 4), 0:3, 0:3)
    @test_throws "offset arrays are not supported" cholesky_crout(A; s = 2)
end

@testitem "cholesky_crout agrees with cholesky when the block size does not divide n" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 10
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))
    for s in (3, 7)
        F = cholesky_crout(A; s)
        L = Matrix(F.L)
        @test norm(L * L' - A) / norm(A) < 1.0e-13
    end
end

@testitem "cholesky_crout agrees with cholesky for a non-BLAS element type" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 6
    B = randn(n, n)
    A = Matrix{BigFloat}(Symmetric(B * B' + n * I))
    # BigFloat is not a BlasReal, so `default_rankk!` falls back to a full `mul!` that writes
    # both triangles of the flushed block instead of only the one named by `uplo`. Agreement here
    # confirms the algorithm still works when the flush leaves the untouched triangle non-zero.
    F = cholesky_crout(A; s = 2)
    L = Matrix(F.L)
    @test norm(L * L' - A) / norm(A) < 1.0e-30
end

@testitem "cholesky_crout calls the rankk! keyword exactly at each flush" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 10
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))
    calls = Ref(0)
    function counting_rankk!(C, block, alpha, beta; uplo::Char = 'L')
        calls[] += 1
        return UpdatableFactorizations.default_rankk!(C, block, alpha, beta; uplo)
    end
    F = cholesky_crout(A; s = 3, rankk! = counting_rankk!)
    # s = 3 on n = 10 flushes after columns 3, 6 and 9 (c = z + s at c = 4, 7, 10).
    @test calls[] == 3
    @test norm(Matrix(F) - A) / norm(A) < 1.0e-13
end
