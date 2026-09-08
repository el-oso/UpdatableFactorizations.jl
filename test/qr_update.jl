@testitem "QR column deletion, every index" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (7, 7), (9, 1))
        A = randn(T, m, n)
        # The fixture must not already be near-orthogonal, or a broken rotation sweep passes.
        @test norm(A' * A - I) > 1
        for j in 1:n
            F = UpdatableQR(A)
            # Poison storage outside the active block before deleting: `delete_column!` must
            # re-establish the zero invariant itself, not rely on it already holding.
            Qbefore = getfield(F, :qrep)
            Rbefore = getfield(F, :factors)
            fill!(view(Rbefore, (F.n + 1):size(Rbefore, 1), :), T(77))
            fill!(view(Rbefore, :, (F.n + 1):size(Rbefore, 2)), T(77))
            fill!(view(Qbefore.buf, :, (F.n + 1):size(Qbefore.buf, 2)), T(88))
            delete_column!(F, j)
            keep = setdiff(1:n, j)
            @test size(F) == (m, n - 1)
            if n > 1
                @test norm(F.Q * F.R - A[:, keep]) / norm(A) < 1.0e-12
                @test norm(F.Q' * F.Q - I) < 1.0e-12
            end
            Q = getfield(F, :qrep)
            R = getfield(F, :factors)
            @test all(iszero, [R[i, j] for j in 1:F.n for i in (j + 1):F.n])
            @test all(iszero, view(Q.buf, :, (F.n + 1):size(Q.buf, 2)))
            @test all(iszero, view(R, (F.n + 1):size(R, 1), :))
            @test all(iszero, view(R, :, (F.n + 1):size(R, 2)))
        end
    end
end

@testitem "QR column deletion rejects an out-of-range index" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    F = UpdatableQR(randn(8, 4))
    @test_throws BoundsError delete_column!(F, 5)
    @test_throws BoundsError delete_column!(F, 0)
end

@testitem "_retriangularize! zeros an injected subdiagonal spike and preserves Q*R" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: _retriangularize!, DenseQ

    Random.seed!(20260908)
    for T in (Float64, ComplexF64)
        m, n = 8, 5
        Qfull = Matrix(qr(randn(T, m, m)).Q)
        R0 = Matrix(triu(randn(T, n, n)))
        R0[3, 2] = randn(T)   # a single subdiagonal spike, as a deleted column leaves behind
        target = Qfull[:, 1:n] * R0

        buf = zeros(T, m, n + 1)
        copyto!(view(buf, :, 1:n), view(Qfull, :, 1:n))
        q = DenseQ{T, Matrix{T}}(buf, m, n)
        Rv = copy(R0)

        _retriangularize!(Rv, q)

        # The subdiagonal must be exactly zero, not merely small: a rounding residue there is
        # the defect this kernel exists to avoid.
        @test all(iszero, [Rv[i, j] for j in 1:n for i in (j + 1):n])
        @test norm(Matrix(q) * Rv - target) / norm(target) < 1.0e-12
        @test norm(Matrix(q)' * Matrix(q) - I) < 1.0e-12
    end
end
