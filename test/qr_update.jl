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

@testitem "QR column shifting, every index pair" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (7, 7))
        A = randn(T, m, n)
        @test norm(A' * A - I) > 1
        for i in 1:n, j in 1:n
            F = UpdatableQR(A)
            shift_columns!(F, i, j)
            p = collect(1:n)
            deleteat!(p, i)
            insert!(p, j, i)
            @test norm(F.Q * F.R - A[:, p]) / norm(A) < 1.0e-12
            @test norm(F.Q' * F.Q - I) < 1.0e-12
            R = getfield(F, :factors)
            @test all(iszero, [R[i, j] for j in 1:F.n for i in (j + 1):F.n])
            @test all(iszero, view(R, :, (n + 1):size(R, 2)))
        end
    end
end

@testitem "QR column shifting re-zeroes poisoned spare storage" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (7, 7))
        A = randn(T, m, n)
        for i in 1:n, j in 1:n
            F = UpdatableQR(A)
            # Poison storage outside the active block before shifting: `shift_columns!` must
            # re-establish the zero invariant itself, not rely on it already holding.
            Qbefore = getfield(F, :qrep)
            Rbefore = getfield(F, :factors)
            fill!(view(Rbefore, (F.n + 1):size(Rbefore, 1), :), T(77))
            fill!(view(Rbefore, :, (F.n + 1):size(Rbefore, 2)), T(77))
            fill!(view(Qbefore.buf, :, (F.n + 1):size(Qbefore.buf, 2)), T(88))
            @test any(!iszero, view(Rbefore, (F.n + 1):size(Rbefore, 1), :))
            @test any(!iszero, view(Rbefore, :, (F.n + 1):size(Rbefore, 2)))
            @test any(!iszero, view(Qbefore.buf, :, (F.n + 1):size(Qbefore.buf, 2)))
            shift_columns!(F, i, j)
            Q = getfield(F, :qrep)
            R = getfield(F, :factors)
            @test all(iszero, view(R, (F.n + 1):size(R, 1), :))
            @test all(iszero, view(R, :, (F.n + 1):size(R, 2)))
            @test all(iszero, view(Q.buf, :, (F.n + 1):size(Q.buf, 2)))
        end
    end
end

@testitem "QR column shifting rejects out-of-range indices" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    F = UpdatableQR(randn(8, 4))
    @test_throws BoundsError shift_columns!(F, 5, 1)
    # `err.a === F` pins the exception to the explicit bounds check rather than an incidental
    # `BoundsError` from indexing into internal storage with the unvalidated argument.
    err = try
        shift_columns!(F, 1, 0)
        nothing
    catch e
        e
    end
    @test err isa BoundsError
    @test err.a === F
    @test iszero(err.i)
end

@testitem "QR rank-1 update" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (8, 7), (6, 6), (9, 1), (30, 12))
        A = randn(T, m, n)
        u = randn(T, m)
        v = randn(T, n)
        F = UpdatableQR(A)
        Qb = copy(F.Q)
        lowrankupdate!(F, u, v)
        @test norm(F.Q * F.R - (A + u * v')) / norm(A) < 1.0e-12
        @test norm(F.Q' * F.Q - I) < 1.0e-12
        R = getfield(F, :factors)
        @test all(iszero, [R[i, j] for j in 1:F.n for i in (j + 1):F.n])
        # A generic `u` has a residual outside the range of `Q`, so the update takes the branch
        # that admits a new direction and the range moves. Measured, this quantity is 0.92 here
        # and 1.5e-15 when the other branch runs. It is the control for the in-range item below.
        if m > n
            @test norm(F.Q - Qb * (Qb' * F.Q)) > 1.0e-6
        end
    end
end

@testitem "QR rank-1 update with an in-range vector" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (6, 6))
        A = randn(T, m, n)
        F = UpdatableQR(A)
        y = randn(T, n)
        u = F.Q * y
        v = randn(T, n)
        Qb = copy(F.Q)
        lowrankupdate!(F, u, v)
        @test norm(F.Q * F.R - (A + u * v')) / norm(A) < 1.0e-12
        @test norm(F.Q' * F.Q - I) < 1.0e-12
        R = getfield(F, :factors)
        @test all(iszero, [R[i, j] for j in 1:F.n for i in (j + 1):F.n])
        # `u` lies in the range of `Q`, so the update stays inside that range: this is what the
        # branch controls, and the reconstruction residual is correct either way. On a square
        # factorization it is the only branch there is.
        @test norm(F.Q - Qb * (Qb' * F.Q)) < 1.0e-12
    end
end

@testitem "QR rank-1 update does not consume its vectors" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    m, n = 9, 5
    F = UpdatableQR(randn(m, n))
    u = randn(m)
    v = randn(n)
    uc = copy(u)
    vc = copy(v)
    lowrankupdate!(F, u, v)
    @test u == uc
    @test v == vc
end

@testitem "QR rank-1 update rejects mismatched vector lengths" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    F = UpdatableQR(randn(9, 5))
    @test_throws "u has length 8, factorization is 9x5" lowrankupdate!(F, zeros(8), zeros(5))
    @test_throws "v has length 4, factorization is 9x5" lowrankupdate!(F, zeros(9), zeros(4))
end

@testitem "QR rank-1 update re-zeroes poisoned spare storage" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    cases = Any[]
    for T in (Float64, ComplexF64)
        A = randn(T, 10, 5)
        push!(cases, (T, A, randn(T, 10), randn(T, 5)))                 # out-of-range branch
        F0 = UpdatableQR(A)
        push!(cases, (T, A, F0.Q * randn(T, 5), randn(T, 5)))           # in-range branch
        Asq = randn(T, 6, 6)
        push!(cases, (T, Asq, randn(T, 6), randn(T, 6)))                # square: no room to grow
        push!(cases, (T, A, zeros(T, 10), randn(T, 5)))                 # zero u
        push!(cases, (T, A, randn(T, 10), zeros(T, 5)))                 # zero v
    end
    for (T, A, u, v) in cases
        F = UpdatableQR(A)
        # Poison storage outside the active block before updating: `lowrankupdate!` must
        # re-establish the zero invariant itself, not rely on it already holding.
        Qbefore = getfield(F, :qrep)
        Rbefore = getfield(F, :factors)
        fill!(view(Rbefore, (F.n + 1):size(Rbefore, 1), :), T(77))
        fill!(view(Rbefore, :, (F.n + 1):size(Rbefore, 2)), T(77))
        fill!(view(Qbefore.buf, :, (F.n + 1):size(Qbefore.buf, 2)), T(88))
        @test any(!iszero, view(Rbefore, (F.n + 1):size(Rbefore, 1), :))
        @test any(!iszero, view(Rbefore, :, (F.n + 1):size(Rbefore, 2)))
        @test any(!iszero, view(Qbefore.buf, :, (F.n + 1):size(Qbefore.buf, 2)))
        lowrankupdate!(F, u, v)
        Q = getfield(F, :qrep)
        R = getfield(F, :factors)
        @test all(iszero, view(R, (F.n + 1):size(R, 1), :))
        @test all(iszero, view(R, :, (F.n + 1):size(R, 2)))
        @test all(iszero, view(Q.buf, :, (F.n + 1):size(Q.buf, 2)))
    end
end

@testitem "_project! computes the two-pass Gram-Schmidt coefficients and residual" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: _project!

    Random.seed!(20260908)
    for T in (Float64, ComplexF64)
        m, n = 10, 4
        Qa = Matrix(qr(randn(T, m, n)).Q)[:, 1:n]
        y = randn(T, n)
        e = randn(T, m)
        e .-= Qa * (Qa' * e)
        e ./= norm(e)
        beta = abs(randn(real(T)))
        r = Qa * y + beta * e
        w = zeros(T, n)
        corr = zeros(T, n)
        rho = _project!(w, r, Qa, corr)
        @test isapprox(w, y; atol = 1.0e-12)
        @test isapprox(rho, beta; atol = 1.0e-12)
        @test isapprox(norm(r), beta; atol = 1.0e-12)
        @test norm(Qa' * r) < 1.0e-12
    end
end

@testitem "_project_residual! removes leakage left by a first pass and accumulates it into w" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: _project_residual!

    Random.seed!(20260908)
    for T in (Float64, ComplexF64)
        m, n = 10, 4
        Qa = Matrix(qr(randn(T, m, n)).Q)[:, 1:n]
        y = randn(T, n)
        w = copy(y)
        r = randn(T, m)
        r .-= Qa * (Qa' * r)
        leak = randn(T, n)
        r .+= Qa * leak
        corr = zeros(T, n)
        rho = _project_residual!(w, r, Qa, corr)
        @test isapprox(w, y .+ leak; atol = 1.0e-10)
        @test norm(Qa' * r) < 1.0e-12
        @test rho == norm(r)
    end
end

@testitem "QR rank-1 update rejects a near-cancelling update by default, admits it at rtol = 0" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64)
        m, n = 10, 5
        A = randn(T, m, n)
        F0 = UpdatableQR(A)
        j = 3
        Rjj = F0.R[j, j]
        delta = 1.0e-10
        # `u` is a scaled copy of `Q`'s own column `j`, so `A + u*v'` keeps every column but `j`
        # unchanged and leaves column `j`'s diagonal contribution shrunk by `delta`: the same
        # `Q` still factors the result, with `R[j,j]` scaled to `delta * Rjj`.
        u = -(1 - delta) * Rjj * F0.Q[:, j]
        v = zeros(T, n)
        v[j] = one(T)

        F = UpdatableQR(A)
        err = try
            lowrankupdate!(F, u, v)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("column $j", err.msg)
        @test occursin("rank deficient", err.msg)
        # The throw left the factorization exactly as it was: it still reconstructs `A`, not
        # `A + u*v'`, `issuccess` holds, and the scratch storage the guard used is clean.
        @test norm(F.Q * F.R - A) / norm(A) < 1.0e-12
        @test issuccess(F)
        Q = getfield(F, :qrep)
        R = getfield(F, :factors)
        @test all(iszero, view(Q.buf, :, (F.n + 1):size(Q.buf, 2)))
        @test all(iszero, view(R, (F.n + 1):size(R, 1), :))
        @test all(iszero, view(R, :, (F.n + 1):size(R, 2)))

        G = UpdatableQR(A)
        lowrankupdate!(G, u, v; rtol = 0.0)
        @test norm(G.Q * G.R - (A + u * v')) / norm(A) < 1.0e-8
        @test norm(G.Q' * G.Q - I) < 1.0e-12
        Rg = getfield(G, :factors)
        @test all(iszero, [Rg[i, k] for k in 1:G.n for i in (k + 1):G.n])
    end
end

@testitem "QR rank-1 update rejects a near-cancelling update on the out-of-range branch" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64)
        m, n = 10, 5
        A = randn(T, m, n)
        F0 = UpdatableQR(A)
        j = 3
        Rjj = F0.R[j, j]
        delta = 1.0e-10
        uin = -(1 - delta) * Rjj * F0.Q[:, j]
        # A genuine out-of-range component of the same tiny magnitude as `delta`: small enough
        # to still trip the guard, but large enough to take the branch that builds a residual
        # direction in `Q`'s augmentation column, so the throw's cleanup has real work to undo.
        e = randn(T, m)
        e .-= F0.Q * (F0.Q' * e)
        e ./= norm(e)
        u = uin + delta .* e
        v = zeros(T, n)
        v[j] = one(T)

        F = UpdatableQR(A)
        err = try
            lowrankupdate!(F, u, v)
            nothing
        catch e2
            e2
        end
        @test err isa ArgumentError
        @test occursin("rank deficient", err.msg)
        @test norm(F.Q * F.R - A) / norm(A) < 1.0e-12
        @test issuccess(F)
        Q = getfield(F, :qrep)
        @test all(iszero, view(Q.buf, :, (F.n + 1):size(Q.buf, 2)))
    end
end

@testitem "QR rank-1 update is unaffected by the default rtol on a generic update" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (6, 6))
        A = randn(T, m, n)
        u = randn(T, m)
        v = randn(T, n)
        F = UpdatableQR(A)
        lowrankupdate!(F, u, v)   # default rtol; must not throw
        @test norm(F.Q * F.R - (A + u * v')) / norm(A) < 1.0e-12
        @test norm(F.Q' * F.Q - I) < 1.0e-12
    end
end

@testitem "QR rank-1 update at rtol = 0 allocates nothing, independent of m" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    n = 30
    for m in (35, 2000)
        A = randn(m, n)
        u = randn(m)
        v = randn(n)
        F = UpdatableQR(A)
        lowrankupdate!(F, u, v; rtol = 0.0)   # warm: compile before measuring
        G = UpdatableQR(A)
        bytes = @allocated lowrankupdate!(G, u, v; rtol = 0.0)
        @test bytes == 0
    end
end
