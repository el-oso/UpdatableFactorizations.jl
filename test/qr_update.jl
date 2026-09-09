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
        qbufbefore = copy(getfield(F, :qrep).buf)
        rbufbefore = copy(getfield(F, :factors))
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
        # `A + u*v'`, `issuccess` holds, and the scratch storage the guard used is clean. The
        # raw buffers are bit-identical to what they were before the call, not merely close.
        @test norm(F.Q * F.R - A) / norm(A) < 1.0e-12
        @test issuccess(F)
        Q = getfield(F, :qrep)
        R = getfield(F, :factors)
        @test Q.buf == qbufbefore
        @test R == rbufbefore
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
        qbufbefore = copy(getfield(F, :qrep).buf)
        rbufbefore = copy(getfield(F, :factors))
        err = try
            lowrankupdate!(F, u, v)
            nothing
        catch e2
            e2
        end
        @test err isa ArgumentError
        @test occursin("rank deficient", err.msg)
        @test norm(F.Q * F.R - A) / norm(A) < 1.0e-12
        # This branch genuinely builds a residual direction in the augmentation column before
        # the guard runs, so a bit-identical raw buffer here confirms the cleanup, not luck.
        @test getfield(F, :qrep).buf == qbufbefore
        @test getfield(F, :factors) == rbufbefore
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

@testitem "QR rank-1 update allocates nothing, at the default rtol and at rtol = 0, independent of m" begin
    using LinearAlgebra, Random, Test

    Random.seed!(20260908)
    n = 30
    for m in (35, 2000)
        A = randn(m, n)
        u = randn(m)
        v = randn(n)

        F = UpdatableQR(A)
        lowrankupdate!(F, u, v; rtol = 0.0)   # warm: compile before measuring
        G = UpdatableQR(A)
        bytes0 = @allocated lowrankupdate!(G, u, v; rtol = 0.0)
        # `@strict` guards `_absorb_spike!`'s call in `lowrankupdate!`: with checks enabled (the
        # default this suite runs under), the guard's own reflection allocates. The kernel is
        # allocation-free with checks disabled, which is the configuration a shipped build sets.
        @test_broken iszero(bytes0)

        H = UpdatableQR(A)
        lowrankupdate!(H, u, v)               # warm the default-rtol path separately
        K = UpdatableQR(A)
        bytesdefault = @allocated lowrankupdate!(K, u, v)
        @test_broken iszero(bytesdefault)
    end
end

@testitem "QR column insertion, every index" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (8, 7), (20, 7), (30, 12))
        Afull = randn(T, m, n + 1)
        for j in 1:(n + 1)
            keep = setdiff(1:(n + 1), j)
            A = Afull[:, keep]
            x = Afull[:, j]
            F = UpdatableQR(A)
            insert_column!(F, j, x)
            @test size(F) == (m, n + 1)
            @test norm(F.Q * F.R - Afull) / norm(Afull) < 1.0e-12
            @test norm(F.Q' * F.Q - I) < 1.0e-12
            R = getfield(F, :factors)
            @test all(iszero, [R[i, j] for j in 1:F.n for i in (j + 1):F.n])
            @test all(iszero, view(R, (F.n + 1):size(R, 1), :))
        end
    end
end

@testitem "QR column insertion grows past its capacity" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity

    Random.seed!(20260908)
    m, n = 10, 4
    Afull = randn(m, n + 1)
    A = Afull[:, 1:n]
    F = UpdatableQR(A; capacity = (m, n))
    @test capacity(F) == (m, n)
    insert_column!(F, n + 1, Afull[:, n + 1])
    @test capacity(F) == (m, 2n)
    @test norm(F.Q * F.R - Afull) / norm(Afull) < 1.0e-12
    @test norm(F.Q' * F.Q - I) < 1.0e-12
end

@testitem "QR column insertion rejects a dependent column" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity

    Random.seed!(20260908)
    m, n = 9, 4
    A = randn(m, n)
    # A tight capacity puts the growth this verb would perform under the same assertion as the
    # rest of its state: a rejected insertion leaves the capacity where it was.
    F = UpdatableQR(A; capacity = (m, n))
    # An exact copy of an existing column leaves a residual of 1.95e-16, which is strictly
    # positive: it is the relative test against `norm(x)` that rejects it, not `rho > 0`.
    @test_throws "lies in the range of the existing columns" insert_column!(F, 2, A[:, 1])
    @test size(F) == (m, n)               # the throw left the factorization as it was
    @test capacity(F) == (m, n)
    @test norm(F.Q * F.R - A) / norm(A) < 1.0e-12
    qb = getfield(F, :qrep).buf
    @test all(iszero, view(qb, :, (F.n + 1):size(qb, 2)))

    # The throw left the factorization exactly as it was: a later, unrelated verb through a
    # different code path succeeds normally right after.
    row = randn(n)
    Afull = vcat(A[1:2, :], row', A[3:end, :])
    insert_row!(F, 3, row)
    @test size(F) == (m + 1, n)
    @test norm(F.Q' * F.Q - I) < 1.0e-12
    @test norm(F.Q * F.R - Afull) / norm(Afull) < 1.0e-12
    Rr = getfield(F, :factors)
    @test all(iszero, [Rr[i, j] for j in 1:F.n for i in (j + 1):F.n])
end

@testitem "QR column insertion rtol widens the dependence test" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    m, n = 9, 4
    A = randn(m, n)
    x = A[:, 1] + 1.0e-6 .* randn(m)
    F = UpdatableQR(A)
    insert_column!(F, n + 1, x)
    @test size(F) == (m, n + 1)
    # The default admits the column, whose residual ratio is 5.9e-7, and leaves a measurably
    # collapsed diagonal entry.
    @test abs(F.R[n + 1, n + 1]) < 1.0e-4

    G = UpdatableQR(A)
    @test_throws "lies in the range of the existing columns" insert_column!(
        G, n + 1, x; rtol = 1.0e-4
    )
end

@testitem "QR column insertion rejects bad indices and a square factorization" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    F = UpdatableQR(randn(9, 4))
    @test_throws BoundsError insert_column!(F, 6, zeros(9))
    @test_throws BoundsError insert_column!(F, 0, zeros(9))
    @test_throws "x has length 8, factorization is 9x4" insert_column!(F, 1, zeros(8))
    G = UpdatableQR(randn(4, 4))
    @test_throws "the factorization requires m >= n" insert_column!(G, 1, zeros(4))
end

@testitem "QR column insertion re-zeroes poisoned spare storage" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    m, n = 9, 4
    for j in (n + 1, 2)   # append, and a position that requires a shift
        A = randn(m, n)
        x = randn(m)
        Afull = j == n + 1 ? hcat(A, x) : hcat(A[:, 1:1], x, A[:, 2:n])
        F = UpdatableQR(A; capacity = (m, 2n))
        # Poison storage outside the active block before inserting: `insert_column!` must
        # re-establish the zero invariant itself, not rely on it already holding. Row and column
        # `F.n + 1` are left alone: that is the augmentation slot the algorithm both reads as a
        # precondition and writes as its result, not dead space it is responsible for clearing.
        Rbefore = getfield(F, :factors)
        Qbefore = getfield(F, :qrep)
        fill!(view(Rbefore, (F.n + 2):size(Rbefore, 1), :), 77.0)
        fill!(view(Rbefore, :, (F.n + 2):size(Rbefore, 2)), 77.0)
        fill!(view(Qbefore.buf, :, (F.n + 2):size(Qbefore.buf, 2)), 88.0)
        @test any(!iszero, view(Rbefore, (F.n + 2):size(Rbefore, 1), :))
        @test any(!iszero, view(Rbefore, :, (F.n + 2):size(Rbefore, 2)))
        @test any(!iszero, view(Qbefore.buf, :, (F.n + 2):size(Qbefore.buf, 2)))
        insert_column!(F, j, x)
        R = getfield(F, :factors)
        Q = getfield(F, :qrep)
        @test all(iszero, view(R, (F.n + 1):size(R, 1), :))
        @test all(iszero, view(R, :, (F.n + 1):size(R, 2)))
        @test all(iszero, view(Q.buf, :, (F.n + 1):size(Q.buf, 2)))
        @test norm(F.Q * F.R - Afull) / norm(Afull) < 1.0e-12
        @test norm(F.Q' * F.Q - I) < 1.0e-12
    end
end

@testitem "QR column insertion allocates nothing when it does not grow, independent of m" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    n = 30
    for m in (35, 2000)
        A = randn(m, n)
        x = randn(m)

        F = UpdatableQR(A)
        insert_column!(F, n + 1, x; rtol = 0.0)   # warm: compile before measuring
        G = UpdatableQR(A)
        bytes0 = @allocated insert_column!(G, n + 1, x; rtol = 0.0)
        @test iszero(bytes0)

        H = UpdatableQR(A)
        insert_column!(H, n + 1, x)               # warm the default-rtol path separately
        K = UpdatableQR(A)
        bytesdefault = @allocated insert_column!(K, n + 1, x)
        @test iszero(bytesdefault)
    end
end

@testitem "QR row insertion, every index" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (8, 7), (6, 6), (9, 1), (30, 12))
        Afull = randn(T, m + 1, n)
        for i in 1:(m + 1)
            keep = setdiff(1:(m + 1), i)
            A = Afull[keep, :]
            x = Afull[i, :]
            F = UpdatableQR(A)
            insert_row!(F, i, x)
            @test size(F) == (m + 1, n)
            @test norm(F.Q * F.R - Afull) / norm(Afull) < 1.0e-12
            @test norm(F.Q' * F.Q - I) < 1.0e-12
            Q = getfield(F, :qrep)
            R = getfield(F, :factors)
            @test all(iszero, [R[i, j] for j in 1:F.n for i in (j + 1):F.n])
            @test all(iszero, view(Q.buf, :, (F.n + 1):size(Q.buf, 2)))
            @test all(iszero, view(R, (F.n + 1):size(R, 1), 1:n))
        end
    end
end

@testitem "QR row insertion grows the row capacity" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: capacity

    Random.seed!(20260908)
    m, n = 8, 4
    Afull = randn(m + 1, n)
    A = Afull[1:m, :]
    F = UpdatableQR(A; capacity = (m, n))
    insert_row!(F, m + 1, Afull[m + 1, :])
    @test capacity(F) == (2m, n)
    @test norm(F.Q * F.R - Afull) / norm(Afull) < 1.0e-12
    @test norm(F.Q' * F.Q - I) < 1.0e-12
end

@testitem "QR row insertion rejects bad indices and lengths" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    F = UpdatableQR(randn(9, 4))
    @test_throws BoundsError insert_row!(F, 11, zeros(4))
    @test_throws BoundsError insert_row!(F, 0, zeros(4))
    @test_throws "x has length 3, factorization is 9x4" insert_row!(F, 1, zeros(3))
end

@testitem "QR row insertion is unaffected by residue in the shared augmentation column" begin
    using LinearAlgebra, Random
    using UpdatableFactorizations: _spare

    Random.seed!(20260908)
    m, n = 8, 4
    i = 3
    Afull = randn(m + 1, n)
    A = Afull[setdiff(1:(m + 1), i), :]
    x = Afull[i, :]
    F = UpdatableQR(A)
    q = getfield(F, :qrep)
    # Poison the column this call builds its unit vector in, with a value distinctive enough
    # that leftover contamination rather than a coincidental zero would show up in the result.
    fill!(_spare(q), 12345.0)
    @test any(!iszero, _spare(q))
    insert_row!(F, i, x)
    @test norm(F.Q * F.R - Afull) / norm(Afull) < 1.0e-12
    @test norm(F.Q' * F.Q - I) < 1.0e-12
    R = getfield(F, :factors)
    @test all(iszero, [R[a, b] for b in 1:F.n for a in (b + 1):F.n])
end

@testitem "QR row deletion, every index" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    for T in (Float64, ComplexF64), (m, n) in ((10, 5), (8, 7), (9, 1), (30, 12))
        A = randn(T, m, n)
        @test norm(A' * A - I) > 1
        for i in 1:m
            F = UpdatableQR(A)
            delete_row!(F, i)
            keep = setdiff(1:m, i)
            @test size(F) == (m - 1, n)
            @test norm(F.Q * F.R - A[keep, :]) / norm(A) < 1.0e-12
            @test norm(F.Q' * F.Q - I) < 1.0e-12
            Q = getfield(F, :qrep)
            R = getfield(F, :factors)
            # The rotation sweep runs k descending so that R never leaves upper-triangular
            # form. An ascending sweep leaves Q*R exact and this subdiagonal at 0.75.
            @test all(iszero, [R[i, j] for j in 1:F.n for i in (j + 1):F.n])
            @test all(iszero, view(Q.buf, (F.m + 1):size(Q.buf, 1), :))
            @test all(iszero, view(Q.buf, :, (F.n + 1):size(Q.buf, 2)))
            @test all(iszero, view(R, (F.n + 1):(F.n + 1), 1:n))
        end
    end
end

@testitem "QR row deletion holds orthogonality across the leverage range" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    m, n = 10, 4
    thrown = 0
    for delta in (1.0e-1, 1.0e-3, 1.0e-5, 1.0e-7, 1.0e-9, 1.0e-12, 1.0e-15, 0.0)
        A = vcat(hcat(randn(m - 1, n - 1), delta .* randn(m - 1)), hcat(randn(1, n - 1), 1.0))
        F = UpdatableQR(A)
        try
            delete_row!(F, m)
        catch err
            err isa ArgumentError || rethrow()
            @test occursin("has leverage one", sprint(showerror, err))
            global thrown += 1
            continue
        end
        @test norm(F.Q' * F.Q - I) < 1.0e-13
        @test norm(F.Q * F.R - A[1:(m - 1), :]) / norm(A) < 1.0e-12
    end
    # Both branches are reached: a sweep that only succeeds, or only throws, tests one of them.
    @test 0 < thrown < 8
end

@testitem "QR row deletion refuses a row of leverage one" begin
    using LinearAlgebra

    A = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    F = UpdatableQR(A)
    @test_throws "row 1 has leverage one" delete_row!(F, 1)
    @test size(F) == (3, 2)               # the throw left the factorization as it was
    @test norm(F.Q * F.R - A) / norm(A) < 1.0e-13
    qb = getfield(F, :qrep).buf
    @test all(iszero, view(qb, :, (F.n + 1):size(qb, 2)))
    delete_row!(F, 3)                     # a row with no leverage deletes cleanly
    @test size(F) == (2, 2)
    @test norm(F.Q * F.R - A[1:2, :]) / norm(A) < 1.0e-13
    @test norm(F.Q' * F.Q - I) < 1.0e-13
end

@testitem "QR row deletion rtol widens the leverage test" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    m, n = 10, 4
    A = vcat(hcat(randn(m - 1, n - 1), 1.0e-6 .* randn(m - 1)), hcat(randn(1, n - 1), 1.0))
    F = UpdatableQR(A)
    delete_row!(F, m)
    @test size(F) == (m - 1, n)
    # The default admits the deletion and leaves a measurably collapsed diagonal entry.
    @test minimum(abs, diag(F.R)) < 1.0e-4

    G = UpdatableQR(A)
    @test_throws "has leverage one" delete_row!(G, m; rtol = 1.0e-4)
end

@testitem "QR row deletion rejects a bad index and a square factorization" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    F = UpdatableQR(randn(9, 4))
    # `err.a === F` and `err.i` pin the exception to the explicit bounds check: indexing into
    # internal storage with the unvalidated argument would raise an incidental `BoundsError`
    # too, on both of these indices, which a bare `@test_throws BoundsError` cannot tell apart.
    err = try
        delete_row!(F, 10)
        nothing
    catch e
        e
    end
    @test err isa BoundsError
    @test err.a === F
    @test err.i == 10
    err0 = try
        delete_row!(F, 0)
        nothing
    catch e
        e
    end
    @test err0 isa BoundsError
    @test err0.a === F
    @test iszero(err0.i)
    G = UpdatableQR(randn(4, 4))
    @test_throws "deleting row 2 would leave a 3x4 factorization" delete_row!(G, 2)
end

@testitem "QR row deletion re-zeroes poisoned spare storage" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    m, n = 9, 4
    i = 3
    A = randn(m, n)
    F = UpdatableQR(A)
    # Poison storage strictly beyond row/column n+1: row n+1 of R is live working space for
    # this verb (it ends the call holding the deleted row of A's coefficients before being
    # re-zeroed), so only what is deeper than that is dead space the call must clear itself.
    # Q's row dimension has no such distinction to poison here: `_deleterow!` owns clearing the
    # single vacated row, and nothing in this verb ever touches a row beyond it.
    Rbefore = getfield(F, :factors)
    Qbefore = getfield(F, :qrep)
    fill!(view(Rbefore, (F.n + 2):size(Rbefore, 1), :), 77.0)
    fill!(view(Rbefore, :, (F.n + 1):size(Rbefore, 2)), 77.0)
    fill!(view(Qbefore.buf, :, (F.n + 1):size(Qbefore.buf, 2)), 88.0)
    @test any(!iszero, view(Rbefore, (F.n + 2):size(Rbefore, 1), :))
    @test any(!iszero, view(Rbefore, :, (F.n + 1):size(Rbefore, 2)))
    @test any(!iszero, view(Qbefore.buf, :, (F.n + 1):size(Qbefore.buf, 2)))
    delete_row!(F, i)
    R = getfield(F, :factors)
    Q = getfield(F, :qrep)
    @test all(iszero, view(R, (F.n + 1):(F.n + 1), 1:n))
    @test all(iszero, view(R, (F.n + 2):size(R, 1), :))
    @test all(iszero, view(R, :, (F.n + 1):size(R, 2)))
    @test all(iszero, view(Q.buf, (F.m + 1):size(Q.buf, 1), :))
    @test all(iszero, view(Q.buf, :, (F.n + 1):size(Q.buf, 2)))
    @test norm(F.Q * F.R - A[setdiff(1:m, i), :]) / norm(A) < 1.0e-12
    @test norm(F.Q' * F.Q - I) < 1.0e-12
end

@testitem "QR row deletion leaves the factorization exactly as it was on a throw" begin
    using LinearAlgebra, Random

    # A row of leverage one built from a generic combination of Q's columns, not a standard
    # basis vector: the residual computed before the throw is then a nontrivial vector at the
    # level of rounding, not an exact zero that would pass whether or not it is cleared.
    Random.seed!(20260908)
    m, n = 10, 4
    A = vcat(hcat(randn(m - 1, n - 1), zeros(m - 1)), hcat(randn(1, n - 1), 1.0))
    F = UpdatableQR(A)
    qbufbefore = copy(getfield(F, :qrep).buf)
    rbufbefore = copy(getfield(F, :factors))
    err = try
        delete_row!(F, m)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("row $m has leverage one", err.msg)
    @test issuccess(F)
    @test getfield(F, :qrep).buf == qbufbefore
    @test getfield(F, :factors) == rbufbefore
end

@testitem "QR row deletion allocates nothing, independent of m" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    n = 15
    for m in (20, 500)
        A = randn(m, n)

        F = UpdatableQR(A)
        delete_row!(F, 3)   # warm: compile before measuring
        G = UpdatableQR(A)
        bytes = @allocated delete_row!(G, 3)
        @test iszero(bytes)
    end
end

@testitem "QR row insertion re-zeroes poisoned spare storage" begin
    using LinearAlgebra, Random

    Random.seed!(20260908)
    m, n = 8, 4
    for cap in ((m, 2n), (2m, 2n))   # growing, then non-growing
        i = 3
        Afull = randn(m + 1, n)
        A = Afull[setdiff(1:(m + 1), i), :]
        x = Afull[i, :]
        F = UpdatableQR(A; capacity = cap)
        # Poison storage beyond what this call reads and writes: row `n + 1` of R at columns
        # `1:n` is live working space here, not dead space the call is responsible for
        # clearing, so it is left alone.
        Rbefore = getfield(F, :factors)
        Qbefore = getfield(F, :qrep)
        fill!(view(Rbefore, (F.n + 2):size(Rbefore, 1), :), 77.0)
        fill!(view(Rbefore, :, (F.n + 1):size(Rbefore, 2)), 77.0)
        fill!(view(Qbefore.buf, :, (F.n + 2):size(Qbefore.buf, 2)), 88.0)
        @test any(!iszero, view(Rbefore, (F.n + 2):size(Rbefore, 1), :))
        @test any(!iszero, view(Rbefore, :, (F.n + 1):size(Rbefore, 2)))
        @test any(!iszero, view(Qbefore.buf, :, (F.n + 2):size(Qbefore.buf, 2)))
        insert_row!(F, i, x)
        R = getfield(F, :factors)
        Q = getfield(F, :qrep)
        @test all(iszero, view(R, (F.n + 1):size(R, 1), :))
        @test all(iszero, view(R, :, (F.n + 1):size(R, 2)))
        @test all(iszero, view(Q.buf, :, (F.n + 1):size(Q.buf, 2)))
        @test norm(F.Q * F.R - Afull) / norm(Afull) < 1.0e-12
        @test norm(F.Q' * F.Q - I) < 1.0e-12
    end
end
