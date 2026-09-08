@testitem "non-BLAS element types" begin
    using LinearAlgebra, ForwardDiff, Random

    Random.seed!(1)
    DualT = ForwardDiff.Dual{Nothing, Float64, 1}

    n = 5
    B64 = randn(n, n)
    A64 = Matrix(Symmetric(B64 * B64' + n * I))
    v64 = randn(n)
    Alu64 = randn(n, n)
    u64 = randn(n)
    w64 = randn(n)
    Bfull64 = randn(n + 1, n + 1)
    Afull64 = Matrix(Symmetric(Bfull64 * Bfull64' + (n + 1) * I))

    # A Dual matrix built from Float64 random entries is positive definite only because the
    # underlying Float64 matrix is; building it directly from random duals would not guarantee
    # that.
    for T in (Float32, BigFloat, DualT)
        A = T.(A64)
        v = T.(v64)
        # BigFloat and Dual carry near-Float64 precision through generic arithmetic, so the
        # generic eps-scaled bound holds comfortably. Float32 rounding is coarser and its
        # error does not scale as cleanly with eps, so its bound is a value measured directly
        # from this test's own data (worst case over the six checks below is ~2e-7).
        tol = T === Float32 ? 1.0e-5 : 100 * sqrt(eps(real(T)))

        F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
        lowrankupdate!(F, v)
        @test norm(Matrix(F) - (A + v * v')) / norm(A) < tol

        G = UpdatableCholesky(cholesky(Symmetric(A + v * v', :L)))
        lowrankdowndate!(G, v)
        @test norm(Matrix(G) - A) / norm(A) < tol

        j = 3
        H = UpdatableCholesky(cholesky(Symmetric(A, :L)))
        delete_column!(H, j)
        keep = setdiff(1:n, j)
        @test norm(Matrix(H) - A[keep, keep]) / norm(A) < tol

        i, k = 1, 4
        S = UpdatableCholesky(cholesky(Symmetric(A, :L)))
        shift_columns!(S, i, k)
        perm = collect(1:n)
        deleteat!(perm, i)
        insert!(perm, k, i)
        @test norm(Matrix(S) - A[perm, perm]) / norm(A) < tol

        Afull = T.(Afull64)
        jc = 2
        keepc = setdiff(1:(n + 1), jc)
        Isub = UpdatableCholesky(cholesky(Symmetric(Afull[keepc, keepc], :L)))
        insert_column!(Isub, jc, Afull[:, jc])
        @test norm(Matrix(Isub) - Afull) / norm(Afull) < tol

        Alu = T.(Alu64)
        u = T.(u64)
        w = T.(w64)
        L = UpdatableLU(lu(Alu, RowMaximum()))
        lowrankupdate!(L, u, w)
        @test norm(Matrix(L) - (Alu + u * w')) / norm(Alu) < tol

        # A Dual built by broadcasting a type constructor over Float64 values carries an
        # all-zero partial, so the checks above pass identically whether or not a kernel
        # actually propagates derivatives. Re-run each operation with one input seeded with a
        # nonzero partial (holding every other input's partial at zero) and check that the
        # result's partial is nonzero and finite. Each case is built so the true derivative is
        # generically nonzero, and each is confirmed (by temporarily discarding the seeded
        # partial before the call) to fail when a kernel silently drops it.
        if T === DualT
            seed(x64, dir) = T.(x64, ForwardDiff.Partials.(tuple.(dir)))
            haspartial(M) = (
                ps = getindex.(ForwardDiff.partials.(M), 1);
                any(!iszero, ps) && all(isfinite, ps)
            )

            dirA = randn(n, n)
            dirv = randn(n)
            dirx = randn(n + 1)
            diru = randn(n)
            dirw = randn(n)

            Az = T.(A64)
            Fp = UpdatableCholesky(cholesky(Symmetric(Az, :L)))
            lowrankupdate!(Fp, seed(v64, dirv))
            @test haspartial(Matrix(Fp))

            Av = seed(A64, dirA)
            v0 = T.(v64)
            Gp = UpdatableCholesky(cholesky(Symmetric(Av + v0 * v0', :L)))
            lowrankdowndate!(Gp, v0)
            @test haspartial(Matrix(Gp))

            Hp = UpdatableCholesky(cholesky(Symmetric(Av, :L)))
            delete_column!(Hp, j)
            @test haspartial(Matrix(Hp))

            Sp = UpdatableCholesky(cholesky(Symmetric(Av, :L)))
            shift_columns!(Sp, i, k)
            @test haspartial(Matrix(Sp))

            Afullz = T.(Afull64)
            Isubp = UpdatableCholesky(cholesky(Symmetric(Afullz[keepc, keepc], :L)))
            insert_column!(Isubp, jc, seed(Afull64[:, jc], dirx))
            @test haspartial(Matrix(Isubp))

            Aluz = T.(Alu64)
            Lp = UpdatableLU(lu(Aluz, RowMaximum()))
            lowrankupdate!(Lp, seed(u64, diru), seed(w64, dirw))
            @test haspartial(Matrix(Lp))
        end
    end
end

@testitem "offset and viewed inputs" begin
    using LinearAlgebra, OffsetArrays, Random

    Random.seed!(2)
    n = 5
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))
    v = randn(n)

    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    lowrankupdate!(F, v)

    G = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    lowrankupdate!(G, OffsetVector(v, 0:(n - 1)))
    @test Matrix(F) ≈ Matrix(G)

    H = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    lowrankupdate!(H, view(vcat(v, randn(3)), 1:n))
    @test Matrix(F) ≈ Matrix(H)

    Ad = A + v * v'
    Fd = UpdatableCholesky(cholesky(Symmetric(Ad, :L)))
    lowrankdowndate!(Fd, v)

    Gd = UpdatableCholesky(cholesky(Symmetric(Ad, :L)))
    lowrankdowndate!(Gd, OffsetVector(v, -2:(n - 3)))
    @test Matrix(Fd) ≈ Matrix(Gd)

    Hd = UpdatableCholesky(cholesky(Symmetric(Ad, :L)))
    lowrankdowndate!(Hd, view(vcat(v, randn(2)), 1:n))
    @test Matrix(Fd) ≈ Matrix(Hd)

    Bfull = randn(n + 1, n + 1)
    Afull = Matrix(Symmetric(Bfull * Bfull' + (n + 1) * I))
    jc = 2
    keepc = setdiff(1:(n + 1), jc)
    Asub = Afull[keepc, keepc]
    xcol = Afull[:, jc]

    Fi = UpdatableCholesky(cholesky(Symmetric(Asub, :L)))
    insert_column!(Fi, jc, xcol)

    Gi = UpdatableCholesky(cholesky(Symmetric(Asub, :L)))
    insert_column!(Gi, jc, OffsetVector(xcol, 0:n))
    @test Matrix(Fi) ≈ Matrix(Gi)

    Hi = UpdatableCholesky(cholesky(Symmetric(Asub, :L)))
    insert_column!(Hi, jc, view(vcat(xcol, randn(3)), 1:(n + 1)))
    @test Matrix(Fi) ≈ Matrix(Hi)

    Alu = randn(n, n)
    u = randn(n)
    w = randn(n)
    L1 = UpdatableLU(lu(Alu, RowMaximum()))
    lowrankupdate!(L1, u, w)

    L2 = UpdatableLU(lu(Alu, RowMaximum()))
    lowrankupdate!(L2, OffsetVector(u, 0:(n - 1)), OffsetVector(w, -1:(n - 2)))
    @test Matrix(L1) ≈ Matrix(L2)

    L3 = UpdatableLU(lu(Alu, RowMaximum()))
    lowrankupdate!(L3, view(vcat(u, randn(2)), 1:n), view(vcat(w, randn(2)), 1:n))
    @test Matrix(L1) ≈ Matrix(L3)
end
