@testitem "UpdatableQR reconstructs the factored matrix" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((9, 5), (6, 6), (8, 7), (9, 1))
        A = randn(T, m, n)
        F = UpdatableQR(A)
        @test size(F) == (m, n)
        @test size(F, 1) == m
        @test size(F, 2) == n
        @test size(F, 3) == 1
        @test norm(F.Q * F.R - A) / norm(A) < 1.0e-13
        @test norm(F.Q' * F.Q - I) < 1.0e-13
        # `F.R` wraps the stored block in `UpperTriangular`, which reports a zero subdiagonal
        # whatever the block holds, so triangularity is asserted on the block itself. Every
        # other item in this milestone does the same, for the same reason.
        Rs = getfield(F, :factors)
        @test all(iszero, [Rs[i, j] for j in 1:F.n for i in (j + 1):F.n])
        @test issuccess(F)
        @test norm(Matrix(F) - A) / norm(A) < 1.0e-13
    end
end

@testitem "UpdatableQR zeroes the reflector storage the standard library leaves behind" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    m, n = 9, 5
    A = randn(m, n)
    G = qr(A)
    # The positive control: LinearAlgebra stores the reflectors below the diagonal of `factors`.
    # Without it this item goes hollow if a future release returns a clean triangle.
    @test any(!iszero, [getfield(G, :factors)[i, j] for j in 1:n for i in (j + 1):m])
    F = UpdatableQR(G)
    R = getfield(F, :factors)
    @test all(iszero, [R[i, j] for j in 1:n for i in (j + 1):size(R, 1)])
    @test all(iszero, view(R, :, (n + 1):size(R, 2)))
    Q = getfield(F, :qrep)
    @test all(iszero, view(Q.buf, (m + 1):size(Q.buf, 1), :))
    @test all(iszero, view(Q.buf, :, (n + 1):size(Q.buf, 2)))
end

@testitem "UpdatableQR exposes live views of its factors" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity

    Random.seed!(20260908)
    F = UpdatableQR(randn(7, 4))
    @test propertynames(F) == (:Q, :R)
    @test :qrep in propertynames(F, true)
    @test :factors in propertynames(F, true)
    Q = F.Q
    Q[1, 1] += 1.0
    @test F.Q[1, 1] == Q[1, 1]        # a view of live storage, not a copy
    @test capacity(F) == (14, 8)
    @test capacity(UpdatableQR(randn(7, 4); capacity = (7, 4))) == (7, 4)
end

@testitem "UpdatableQR rejects inconsistent construction arguments" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    A = randn(6, 3)
    @test_throws "row capacity 5 is below the size 6" UpdatableQR(A; capacity = (5, 3))
    @test_throws "column capacity 2 is below the size 3" UpdatableQR(A; capacity = (6, 2))
    @test_throws "it requires m >= n" UpdatableQR(qr(randn(3, 6)))
    @test_throws "dimension must be positive" size(UpdatableQR(A), 0)
    # The guard is an exact-zero test on the diagonal, so the fixture makes a diagonal entry
    # exactly zero. Householder QR of a matrix that is only numerically rank deficient does
    # not: for `B[:, 3] = B[:, 1]` the third diagonal entry measures 1.3e-16, and the
    # factorization is accepted. `insert_column!`'s `rtol` is the relative test.
    C = randn(6, 3)
    C[:, 2] .= 0.0
    @test_throws "is rank deficient" UpdatableQR(C)
end

@testitem "UpdatableQR constructs from a non-BLAS factorization" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    A = BigFloat.(randn(7, 4))
    G = qr(A)
    @test G isa LinearAlgebra.QR          # not QRCompactWY; the constructor takes both
    F = UpdatableQR(G)
    @test norm(F.Q * F.R - A) / norm(A) < 1.0e-30
end

@testitem "UpdatableQR wraps an already-computed thin factorization" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((9, 5), (6, 6))
        A = randn(T, m, n)
        G = qr(A)
        F = UpdatableQR(Matrix(G.Q), Matrix(G.R); capacity = (m, n))
        @test capacity(F) == (m, n)
        @test norm(Matrix(F) - A) / norm(A) < 1.0e-13
        @test norm(F.Q' * F.Q - I) < 1.0e-13
    end
    @test_throws "Q is 6 by 4 and R is 3 by 3" UpdatableQR(randn(6, 4), randn(3, 3))
    @test_throws "R is 4 by 3; it must be square" UpdatableQR(randn(6, 4), randn(4, 3))
end

@testitem "qr_householder factors through the standard library" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((9, 5), (6, 6), (9, 1))
        A = randn(T, m, n)
        F = qr_householder(A)
        @test F isa UpdatableQR
        @test norm(Matrix(F) - A) / norm(A) < 1.0e-13
        # The accuracy claim the docs make for this path: orthogonality at machine precision,
        # unconditionally, because the factorization is LinearAlgebra's.
        @test norm(F.Q' * F.Q - I) < 10 * eps(real(T)) * n
    end
    @test capacity(qr_householder(randn(9, 5); capacity = (9, 5))) == (9, 5)
end
