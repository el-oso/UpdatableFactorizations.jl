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

@testitem "qr_bcgs factors a rectangular matrix" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity
    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((12, 7), (9, 9)), s in (1, 3, 64)
        A = randn(T, m, n)
        F = qr_bcgs(A; s)
        @test F isa UpdatableQR{T}
        @test size(F) == (m, n)
        @test size(F.Q) == (m, n)
        @test norm(Matrix(F) - A) / norm(A) < 1.0e-12
        @test norm(F.Q' * F.Q - I) < 1.0e-12
        # `F.R` wraps the stored block in `UpperTriangular`, which reports a zero subdiagonal
        # whatever the block holds, so triangularity is asserted on the block itself.
        Rs = getfield(F, :factors)
        @test all(iszero, [Rs[i, j] for j in 1:n for i in (j + 1):n])
    end
    # The factorization is updatable on return, since qr_bcgs returns an UpdatableQR.
    A = randn(12, 7)
    F = qr_bcgs(A; s = 4)
    x = randn(12)
    insert_column!(F, 8, x)
    @test norm(Matrix(F) - [A x]) / norm([A x]) < 1.0e-12
    @test capacity(qr_bcgs(A; s = 4, capacity = (12, 7))) == (12, 7)
end

@testitem "qr_bcgs reorthogonalization recovers orthogonality" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 12
    # A Vandermonde matrix of condition number 8.8e8: block CGS alone loses orthogonality as the
    # square of the condition number, reaching 2.3e-4 here, and one reorthogonalization pass
    # brings it back to 2.3e-12 because cond(A) * eps is still far below one, which is the
    # regime the bound requires.
    A = [x^(j - 1) for x in range(0.0, 1.0; length = n), j in 1:n]
    Qn = qr_bcgs(A; s = 8, reorth = false).Q
    Qy = qr_bcgs(A; s = 8, reorth = true).Q
    @test norm(Qn' * Qn - I) > 1.0e-6
    @test norm(Qy' * Qy - I) < 1.0e-8
end

@testitem "qr_bcgs rejects a wide matrix" begin
    using LinearAlgebra
    @test_throws "is 3 by 5; qr_bcgs requires m >= n" qr_bcgs(randn(3, 5); s = 2)
end

@testitem "qr_bcgs rejects a rank-deficient column" begin
    using LinearAlgebra
    A = [1.0 0.0 1.0; 0.0 1.0 0.0; 0.0 0.0 0.0]
    @test_throws "column 3 is a combination of the columns before it" qr_bcgs(A; s = 2)
    # A column that is dependent only up to rounding leaves noise rather than an exact zero, so
    # the default exact-zero test accepts it and `rtol` is what catches it.
    B = [1.0 0.0 1.0; 0.0 1.0 1.0e-15; 0.0 0.0 1.0e-17]
    @test size(qr_bcgs(B; s = 2)) == (3, 3)
    @test_throws "column 3 is a combination of the columns before it" qr_bcgs(B; s = 2, rtol = 1.0e-8)
end

@testitem "qr_bcgs rejects an offset matrix" begin
    using LinearAlgebra, OffsetArrays
    A = OffsetMatrix(randn(4, 3), 0:3, 0:2)
    @test_throws "offset arrays are not supported" qr_bcgs(A; s = 2)
end

@testitem "qr_bcgs block size divides, does not divide, and covers the whole matrix" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 12
    A = randn(n, n)
    Aref = qr(A)
    for s in (1, n, 5)   # s = 1: every column flushes; s = n: one block, never flushes
        F = qr_bcgs(A; s)
        @test norm(Matrix(F) - A) / norm(A) < 1.0e-12
        @test norm(F.Q' * F.Q - I) < 1.0e-12
        # Compared against a fresh `qr` of the same matrix rather than against algebra derived
        # from `F` itself, so an implementation bug in `qr_bcgs` cannot cancel against the check.
        @test norm(abs.(diag(F.R)) - abs.(diag(Aref.R))) / norm(diag(Aref.R)) < 1.0e-10
    end
end

@testitem "qr_bcgs returns a factorization on which the updating verbs work" begin
    using LinearAlgebra, Random, Test
    using UpdatableFactorizations: TypeContracts
    using .TypeContracts: behavior_passes
    Random.seed!(20260908)
    m, n = 10, 6
    A = randn(m, n)
    F = qr_bcgs(A; s = 3)
    Rs = getfield(F, :factors)
    @test iszero(norm(tril(Rs[1:n, 1:n], -1)))
    lowrankupdate!(F, randn(m), randn(n))
    @test norm(Matrix(F) - A) > 1.0e-10   # the update actually changed the factorization
    delete_column!(F, 2)
    @test size(F) == (m, n - 1)
    @test norm(F.Q' * F.Q - I) < 1.0e-10
    @test iszero(norm(tril(getfield(F, :factors)[1:(n - 1), 1:(n - 1)], -1)))
    @test behavior_passes(UpdatableQR, [F])
end

@testitem "qr_bcgs conditioning sweep: reorth = true stays orthogonal, false does not" begin
    using LinearAlgebra, Printf
    # A Vandermonde matrix on n equally spaced nodes in [0, 1]: cond(A) rises by roughly two
    # orders of magnitude per two columns, which sweeps kappa across ten decades while s stays
    # fixed, so the sweep isolates the effect of conditioning from the effect of block size.
    println("qr_bcgs orthogonality vs conditioning (s = 8):")
    @printf("%3s %12s %14s %14s\n", "n", "cond(A)", "reorth=false", "reorth=true")
    for n in 6:2:18
        A = [x^(j - 1) for x in range(0.0, 1.0; length = n), j in 1:n]
        kappa = cond(A)
        en = norm(qr_bcgs(A; s = 8, reorth = false).Q' * qr_bcgs(A; s = 8, reorth = false).Q - I)
        ey = norm(qr_bcgs(A; s = 8, reorth = true).Q' * qr_bcgs(A; s = 8, reorth = true).Q - I)
        @printf("%3d %12.2e %14.2e %14.2e\n", n, kappa, en, ey)
        # The reorthogonalization pass must never do worse than skipping it, at every point on
        # the sweep including where both have already broken down.
        @test ey <= en
        # Below this threshold, kappa * eps is comfortably under one, which is the regime the
        # Giraud-Langou-Rozloznik bound requires; here reorth = true stays within a few orders of
        # magnitude of eps while reorth = false has already lost 2 to 9 digits of orthogonality.
        if kappa < 1.0e11
            @test ey < 1.0e-8
            @test en > 1.0e-14
        end
    end
end

@testitem "the flush keyword arguments are the functions that run" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 12
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))

    calls = Ref(0)
    counting_rankk!(C, X, alpha, beta; uplo = 'L') =
        (calls[] += 1; UpdatableFactorizations.default_rankk!(C, X, alpha, beta; uplo))
    F = cholesky_crout(A; s = 4, rankk! = counting_rankk!)
    @test calls[] == 2                       # flushes at columns 5 and 9
    @test norm(Matrix(F) - A) / norm(A) < 1.0e-13

    calls[] = 0
    counting_matmul!(C, X, Y, alpha, beta) = (calls[] += 1; mul!(C, X, Y, alpha, beta))
    A2 = randn(n, n)
    G = lu_crout(A2; s = 4, matmul! = counting_matmul!)
    # Two flushes, nine deferred column updates and eight deferred row updates. `Matrix(G)` is
    # rebuilt from `G.L` and `G.U`, so comparing against it would compare the factorization
    # with itself.
    @test calls[] == 19
    @test norm(G.L * G.U - A2[G.p, :]) / norm(A2) < 1.0e-12
end

@testitem "a full-block flush gives the same Cholesky factor as a rank-k flush" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 12
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))
    gemm_rankk!(C, X, alpha, beta; uplo = 'L') = mul!(C, X, X', alpha, beta)
    F = cholesky_crout(A; s = 4)
    G = cholesky_crout(A; s = 4, rankk! = gemm_rankk!)
    @test norm(Matrix(F.L) - Matrix(G.L)) / norm(Matrix(G.L)) < 1.0e-13
end

@testitem "qr_bcgs calls the matmul! keyword at each flush and never at s >= n" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    m, n = 16, 12
    A = randn(m, n)

    calls = Ref(0)
    counting_matmul!(C, X, Y, alpha, beta) = (calls[] += 1; mul!(C, X, Y, alpha, beta))

    # s = 4 on n = 12 flushes at columns 5 and 9 (c = z + s), since z advances to c at each
    # flush. R is accumulated from the coefficients each flush and each column already compute,
    # so `matmul!` is called only within the flush: 2 calls per flush, 4 with reorth since the
    # projection runs twice.
    F = qr_bcgs(A; s = 4, matmul! = counting_matmul!)
    @test calls[] == 8
    @test norm(Matrix(F) - A) / norm(A) < 1.0e-12

    calls[] = 0
    Fn = qr_bcgs(A; s = 4, reorth = false, matmul! = counting_matmul!)
    @test calls[] == 4
    @test norm(Matrix(Fn) - A) / norm(A) < 1.0e-12

    # s = n never satisfies c == z + s for c in 1:n, so the flush branch is never taken, and
    # `matmul!` is not called at all: the factorization is still correct with the substitute
    # never invoked.
    calls[] = 0
    Fu = qr_bcgs(A; s = n, matmul! = counting_matmul!)
    @test iszero(calls[])
    @test norm(Matrix(Fu) - A) / norm(A) < 1.0e-12
end

@testitem "the flush keyword arguments run identically on complex data" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 12
    B = randn(ComplexF64, n, n)
    A = Matrix(Hermitian(B * B' + n * I))
    calls = Ref(0)
    counting_rankk!(C, X, alpha, beta; uplo = 'L') =
        (calls[] += 1; UpdatableFactorizations.default_rankk!(C, X, alpha, beta; uplo))
    # A substitute that only forwards to the default must reproduce it exactly, not merely to
    # within a tolerance, because it runs the identical arithmetic in the identical order.
    F = cholesky_crout(A; s = 4, rankk! = counting_rankk!)
    Fd = cholesky_crout(A; s = 4)
    @test calls[] == 2
    @test Matrix(F.L) == Matrix(Fd.L)

    calls[] = 0
    counting_matmul!(C, X, Y, alpha, beta) = (calls[] += 1; mul!(C, X, Y, alpha, beta))
    A2 = randn(ComplexF64, n, n)
    G = lu_crout(A2; s = 4, matmul! = counting_matmul!)
    Gd = lu_crout(A2; s = 4)
    @test calls[] == 19
    @test G.L == Gd.L && G.U == Gd.U && G.p == Gd.p

    calls[] = 0
    m = 16
    A3 = randn(ComplexF64, m, n)
    Qs = qr_bcgs(A3; s = 4, matmul! = counting_matmul!)
    Qd = qr_bcgs(A3; s = 4)
    @test calls[] == 8
    @test Qs.Q == Qd.Q && getfield(Qs, :factors) == getfield(Qd, :factors)
end

@testitem "substituting a flush that computes the wrong update corrupts each factorization" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    n = 12
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))
    # Doubling the correction term is wrong at every flush, and s = 4 on n = 12 flushes twice
    # (at columns 5 and 9), so this configuration reaches the corrupted code path.
    wrong_rankk!(C, X, alpha, beta; uplo = 'L') =
        UpdatableFactorizations.default_rankk!(C, X, 2 * alpha, beta; uplo)
    Fw = cholesky_crout(A; s = 4, rankk! = wrong_rankk!)
    @test norm(Matrix(Fw) - A) / norm(A) > 1.0e-3

    Random.seed!(20260908)
    A2 = randn(n, n)
    wrong_matmul!(C, X, Y, alpha, beta) = mul!(C, X, Y, 2 * alpha, beta)
    Gw = lu_crout(A2; s = 4, matmul! = wrong_matmul!)
    @test norm(Gw.L * Gw.U - A2[Gw.p, :]) / norm(A2) > 1.0e-3

    Random.seed!(20260908)
    m = 16
    A3 = randn(m, n)
    Qw = qr_bcgs(A3; s = 4, matmul! = wrong_matmul!)
    @test norm(Matrix(Qw) - A3) / norm(A3) > 1.0e-3
end
