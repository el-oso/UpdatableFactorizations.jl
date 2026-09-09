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
