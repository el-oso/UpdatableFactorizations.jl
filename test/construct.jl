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

@testitem "default_rankk! rejects a non-real alpha on a complex element type" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    A = randn(ComplexF64, 9, 4)
    C = Matrix(Hermitian(randn(ComplexF64, 9, 9)))
    @test_throws InexactError UpdatableFactorizations.default_rankk!(C, A, 2.0 + 0.7im, 1.0)
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

@testitem "lu_crout without pivoting factors" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 9
    for T in (Float64, ComplexF64), s in (1, 4, 64)
        A = randn(T, n, n) + n * I
        F = lu_crout(A; s, pivot = NoPivot())
        # With no pivoting the permutation must be the identity, which is what makes the
        # residual below a test of the factorization rather than of the permutation.
        @test F.p == 1:n
        @test norm(F.L * F.U - A) / norm(A) < 1.0e-12
        @test norm(A * (F \ ones(T, n)) - ones(T, n)) < 1.0e-10
    end
end

@testitem "lu_crout without pivoting throws on a zero pivot" begin
    using LinearAlgebra
    A = [1.0 2.0 3.0; 2.0 4.0 5.0; 1.0 1.0 1.0]   # the leading 2x2 minor is singular
    # The second pivot is the one that vanishes, and the exception names it.
    @test_throws ZeroPivotException(2) lu_crout(A; s = 2, pivot = NoPivot())
end

@testitem "lu_crout rtol widens the zero-pivot test" begin
    using LinearAlgebra
    A = [1.0 2.0 3.0; 2.0 (4.0 + 1.0e-13) 5.0; 1.0 1.0 1.0]
    F = lu_crout(A; s = 2, pivot = NoPivot())
    @test issuccess(F)
    @test abs(F.d[2]) < 1.0e-12
    @test_throws ZeroPivotException(2) lu_crout(A; s = 2, pivot = NoPivot(), rtol = 1.0e-8)
end

@testitem "lu_crout rejects an unsupported pivoting strategy" begin
    using LinearAlgebra
    A = Matrix(1.0I, 4, 4)
    @test_throws "use NoPivot() or RowMaximum()" lu_crout(A; s = 2, pivot = ColumnNorm())
end

@testitem "lu_crout rejects a block size below one" begin
    using LinearAlgebra
    A = Matrix(1.0I, 4, 4)
    @test_throws "block size s must be at least 1, got 0" lu_crout(A; s = 0)
end

@testitem "lu_crout rejects an offset matrix" begin
    using LinearAlgebra, OffsetArrays
    A = OffsetMatrix(Matrix(1.0I, 4, 4), 0:3, 0:3)
    @test_throws "offset arrays are not supported" lu_crout(A; s = 2)
end

@testitem "lu_crout with the default pivoting strategy runs and reconstructs" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 9
    for T in (Float64, ComplexF64), s in (1, 4, 9, 64)
        A = randn(T, n, n)
        F = lu_crout(A; s)   # pivot = RowMaximum() by default
        @test sort(F.p) == 1:n
        @test norm(F.L * F.U - A[F.p, :]) / norm(A) < 1.0e-12
        @test norm(A * (F \ ones(T, n)) - ones(T, n)) < 1.0e-9
    end
end

@testitem "lu_crout calls the matmul! keyword at each flush" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 10
    A = randn(n, n) + n * I
    # matmul! also carries the per-column trailing update (a matrix-vector call), so only the
    # matrix-shaped calls -- C a matrix rather than a vector -- are the periodic block flush.
    flush_calls = Ref(0)
    function counting_matmul!(C, Bl, Br, alpha, beta)
        ndims(C) == 2 && (flush_calls[] += 1)
        return mul!(C, Bl, Br, alpha, beta)
    end
    F = lu_crout(A; s = 3, pivot = NoPivot(), matmul! = counting_matmul!)
    # s = 3 on n = 10 flushes after columns 3, 6 and 9 (c = z + s at c = 4, 7, 10).
    @test flush_calls[] == 3
    @test norm(F.L * F.U - A) / norm(A) < 1.0e-12
end

@testitem "lu_crout agrees with lu when the block size does not divide n" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 10
    for s in (3, 7)
        A = randn(n, n) + n * I
        F = lu_crout(A; s, pivot = NoPivot())
        @test norm(F.L * F.U - A) / norm(A) < 1.0e-12
    end
end

@testitem "lu_crout returns a factorization that supports lowrankupdate!" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 8
    A = randn(n, n) + n * I
    F = lu_crout(A; s = 3, pivot = NoPivot())
    u = randn(n) / n
    v = randn(n) / n
    lowrankupdate!(F, u, v)
    @test issuccess(F)
    @test norm(Matrix(F) - (A + u * v')) / norm(A) < 1.0e-10
    @test norm((A + u * v') * (F \ ones(n)) - ones(n)) < 1.0e-8
end

@testitem "lu_crout with partial pivoting factors" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 9
    for T in (Float64, ComplexF64), s in (1, 4, 9, 64)
        A = randn(T, n, n)
        F = lu_crout(A; s)
        # A plain random matrix must produce a nontrivial permutation, or the residual below
        # would pass for an implementation that ignores the pivot entirely.
        @test F.p != 1:n
        @test norm(F.L * F.U - A[F.p, :]) / norm(A) < 1.0e-12
        @test norm(A * (F \ ones(T, n)) - ones(T, n)) < 1.0e-10
    end
end

@testitem "lu_crout with partial pivoting matches the standard library" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 12
    for s in (1, 4, 5, 12)
        A = randn(n, n)
        F = lu_crout(A; s)
        G = lu(A)
        @test F.p != 1:n
        # Partial pivoting picks the same rows in the same order, so the permutations agree
        # exactly, across a block size that divides n (4), one that does not (5), one flush per
        # column (1) and one that never flushes at all (12 = n).
        @test F.p == G.p
        @test norm(Matrix(F.L) - Matrix(G.L)) / norm(Matrix(G.L)) < 1.0e-12
        @test norm(Matrix(F.U) - Matrix(G.U)) / norm(Matrix(G.U)) < 1.0e-12
    end
end

@testitem "lu_crout with partial pivoting matches the reference algorithm on complex data" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 12
    for s in (1, 4, 5, 12)
        A = randn(ComplexF64, n, n)
        F = lu_crout(A; s)
        # LAPACK's zgetrf picks the pivot by |Re|+|Im| (IZAMAX), not by magnitude, so `lu(A)` is
        # not the right reference for a complex comparison here. `generic_lufact!` is Julia's own
        # non-LAPACK partial-pivoting algorithm and picks by magnitude, the same quantity
        # `_pivotrow` compares, so it agrees with `lu_crout` exactly.
        G = LinearAlgebra.generic_lufact!(copy(A))
        @test F.p != 1:n
        @test F.p == G.p
        @test norm(Matrix(F.L) - Matrix(G.L)) / norm(Matrix(G.L)) < 1.0e-12
        @test norm(Matrix(F.U) - Matrix(G.U)) / norm(Matrix(G.U)) < 1.0e-12
    end
end

@testitem "lu_crout survives a matrix whose unpivoted factorization does not exist" begin
    using LinearAlgebra
    A = [0.0 1.0; 1.0 0.0]
    @test_throws ZeroPivotException(1) lu_crout(A; s = 1, pivot = NoPivot())
    F = lu_crout(A; s = 1)
    @test F.p == [2, 1]
    @test norm(F.L * F.U - A[F.p, :]) < 1.0e-14
end

@testitem "lu_crout with partial pivoting supports lowrankupdate!" begin
    using LinearAlgebra, Random, Test
    using UpdatableFactorizations: TypeContracts
    using .TypeContracts: behavior_passes
    Random.seed!(20260908)
    n = 8
    A = randn(n, n)
    F = lu_crout(A; s = 3)
    @test F.p != 1:n
    u = randn(n) / n
    v = randn(n) / n
    lowrankupdate!(F, u, v)
    @test issuccess(F)
    @test norm(Matrix(F) - (A + u * v')) / norm(A) < 1.0e-9
    @test norm((A + u * v') * (F \ ones(n)) - ones(n)) < 1.0e-7
    @test behavior_passes(UpdatableLU, [F])
end
