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
        # whatever the block holds, so triangularity is asserted on the block itself.
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
    R = F.R
    R[1, 2] += 1.0
    @test F.R[1, 2] == R[1, 2]        # a view of live storage, not a copy
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

@testitem "_grow! grows storage without losing or leaking data" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity, _grow!

    Random.seed!(20260908)
    m, n = 6, 3
    mcap, ncap = m + 2, n + 2   # strictly larger than the size, so there is real dead space
    #                             between the active block and the buffer edge to poison
    A = randn(m, n)

    for (mneeded, nneeded, growsn) in ((m + 5, n, false), (m, n + 5, true), (m + 5, n + 5, true))
        F = UpdatableQR(A; capacity = (mcap, ncap))
        q = getfield(F, :qrep)
        oldqbuf = q.buf
        Rbefore = getfield(F, :factors)
        # Poison the dead space beyond the active block and its single augmentation
        # column/row: a correct `_grow!` never reads from there, so if a broken copy read
        # too wide, this leaks into the regrown buffer instead of its fresh zero fill.
        fill!(view(oldqbuf, (m + 1):mcap, :), 99.0)
        fill!(view(oldqbuf, :, (n + 2):(ncap + 1)), 99.0)
        fill!(view(Rbefore, (n + 1):(ncap + 1), :), 99.0)
        fill!(view(Rbefore, :, (n + 1):(ncap + 1)), 99.0)

        _grow!(F, mneeded, nneeded)
        newmcap, newncap = capacity(F)
        @test newmcap >= mneeded && newncap >= nneeded

        @test norm(F.Q * F.R - A) / norm(A) < 1.0e-13
        @test norm(F.Q' * F.Q - I) < 1.0e-13

        @test q.buf !== oldqbuf   # Q's buffer reallocates whenever `_grow!` proceeds
        @test all(iszero, view(q.buf, :, F.n + 1))
        @test all(iszero, view(q.buf, (F.m + 1):newmcap, :))
        @test all(iszero, view(q.buf, :, (F.n + 2):(newncap + 1)))

        R = getfield(F, :factors)
        if growsn
            @test R !== Rbefore
            @test all(iszero, view(R, (F.n + 1):(newncap + 1), :))
            @test all(iszero, view(R, :, (F.n + 1):(newncap + 1)))
        else
            # Growing rows alone must not disturb R: same object, poison untouched.
            @test R === Rbefore
            @test all(==(99.0), view(R, (n + 1):(ncap + 1), :))
        end
    end
end

@testitem "UpdatableQR solves least squares problems" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((9, 5), (6, 6), (9, 1))
        A = randn(T, m, n)
        b = randn(T, m)
        F = UpdatableQR(A)
        x = F \ b
        # A least-squares solution has n entries; the generic Factorization fallback returns m.
        @test length(x) == n
        @test norm(A' * (A * x - b)) / norm(A' * b) < 1.0e-11

        y = similar(b, n)
        ldiv!(y, F, b)
        @test y ≈ x

        B = randn(T, m, 3)
        X = F \ B
        @test size(X) == (n, 3)
        @test norm(A' * (A * X - B)) / norm(A' * B) < 1.0e-11

        C = copy(B)
        ldiv!(F, C)
        @test C[1:n, :] ≈ X
    end
end

@testitem "UpdatableQR solving rejects mismatched shapes and offset arguments" begin
    using LinearAlgebra, OffsetArrays, Random

    Random.seed!(20260908)
    F = UpdatableQR(randn(9, 5))
    @test_throws "b has length 4, factorization is 9x5" ldiv!(zeros(5), F, zeros(4))
    @test_throws "y has length 4, factorization is 9x5" ldiv!(zeros(4), F, zeros(9))
    @test_throws "B has 4 rows, factorization is 9x5" ldiv!(F, zeros(4, 2))
    # The solve path is one-based by declaration, unlike the updating verbs.
    @test_throws "offset arrays are not supported" ldiv!(
        zeros(5), F, OffsetVector(zeros(9), 0:8)
    )
    @test_throws "offset arrays are not supported" F \ OffsetVector(zeros(9), 0:8)
    @test_throws "B has 4 rows, factorization is 9x5" F \ zeros(4)
    @test_throws "B has 4 rows, factorization is 9x5" F \ zeros(4, 2)
end

@testitem "UpdatableQR solving promotes element types without introducing ambiguity" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    A = randn(9, 5)
    F = UpdatableQR(A)
    b = randn(9)

    # An integer right-hand side is solved at the promoted (floating) type, not truncated to
    # Int: a naive `eltype(F)`-only override would reintroduce this regression.
    bi = rand(-9:9, 9)
    @test F \ bi ≈ F \ Float64.(bi)
    @test eltype(F \ bi) == Float64

    # A complex right-hand side against a real factorization must not be an ambiguous method:
    # LinearAlgebra's own real/complex reinterpretation trick applies to the same argument
    # types, and neither method is strictly more specific than the other.
    bc = randn(ComplexF64, 9)
    x = F \ bc
    @test length(x) == 5
    @test x ≈ (A \ bc)
    Bc = randn(ComplexF64, 9, 3)
    @test (F \ Bc) ≈ (A \ Bc)

    # Agreement with LinearAlgebra's own solve, overdetermined and square.
    @test (F \ b) ≈ (A \ b)
    Asq = randn(6, 6)
    Fsq = UpdatableQR(Asq)
    bsq = randn(6)
    @test (Fsq \ bsq) ≈ (Asq \ bsq)
end
