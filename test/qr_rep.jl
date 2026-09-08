@testitem "DenseQ implements the AbstractQRep contract" begin
    using LinearAlgebra
    using UpdatableFactorizations: AbstractQRep, DenseQ, materialize, TypeContracts
    using UpdatableFactorizations.TypeContracts: @test_implements
    @test_implements DenseQ AbstractQRep
end

@testitem "DenseQ presents its active block and keeps the rest zero" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: DenseQ, materialize, _active, _spare, capacity

    Random.seed!(20260908)
    m, n = 6, 3
    buf = zeros(6, 5)
    A = qr(randn(m, n)).Q * Matrix(I, m, n)
    copyto!(view(buf, 1:m, 1:n), A)
    q = DenseQ{Float64,Matrix{Float64}}(buf, m, n)

    @test size(q) == (m, n)
    @test size(q, 1) == m
    @test size(q, 3) == 1
    @test_throws "dimension must be positive" size(q, 0)
    @test eltype(q) === Float64
    @test Matrix(q) == A
    @test materialize(q) === q
    @test all(iszero, _spare(q))
    @test capacity(q) == (6, 4)
end

@testitem "DenseQ applies rotations to the augmented factor" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: DenseQ, _active, _augmented

    Random.seed!(20260908)
    for T in (Float64, ComplexF64)
        m, n = 7, 4
        buf = zeros(T, m, n + 1)
        copyto!(view(buf, 1:m, 1:n), qr(randn(T, m, n)).Q * Matrix{T}(I, m, n))
        q = DenseQ{T,Matrix{T}}(buf, m, n)
        c, s, _ = LinearAlgebra.givensAlgorithm(one(T), one(T))
        G = LinearAlgebra.Givens(2, 3, T(c), T(s))

        # Orthonormality alone is preserved by doing nothing, so each side is checked against
        # the rotation written out as a matrix. `Matrix(::Givens, n)` does not exist; rotating
        # an identity of the right size is how the matrix is formed.
        Gright = rmul!(Matrix{T}(I, n + 1, n + 1), G)
        Gleft = lmul!(G, Matrix{T}(I, m, m))

        ref = copy(_augmented(q))
        rmul!(q, G)
        @test _augmented(q) ≈ ref * Gright
        @test norm(_active(q)' * _active(q) - I) < 1.0e-14

        ref = copy(_augmented(q))
        lmul!(G, q)
        @test _augmented(q) ≈ Gleft * ref
        @test norm(_active(q)' * _active(q) - I) < 1.0e-14
    end
end

@testitem "DenseQ structural edits leave storage outside the active block zero" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: DenseQ, _insertrow!, _deleterow!, _dropcolumn!, _spare

    Random.seed!(20260908)
    m, n = 5, 3
    # Every entry starts nonzero, including the region a mutator must clear itself: nothing
    # here should incidentally start at zero and mask a mutator that never wrote to it.
    buf = fill(99.0, m + 1, n + 1)
    B = randn(m, n)
    copyto!(view(buf, 1:m, 1:n), B)
    q = DenseQ{Float64,Matrix{Float64}}(buf, m, n)

    _insertrow!(q, 2)
    @test size(q) == (m + 1, n)
    @test view(q.buf, 2, 1:n) == zeros(n)
    @test view(q.buf, 3:(m+1), 1:n) == B[2:m, :]
    @test all(iszero, view(q.buf, :, n + 1))

    _deleterow!(q, 2)
    @test size(q) == (m, n)
    @test view(q.buf, 1:m, 1:n) == B
    @test all(iszero, view(q.buf, (m+1):(m+1), :))
    @test all(iszero, _spare(q))

    fill!(view(q.buf, :, n + 1), 99.0) # poison the augmentation column again before dropping a column
    _dropcolumn!(q)
    @test size(q) == (m, n - 1)
    @test all(iszero, view(q.buf, :, n:(n+1)))
end
