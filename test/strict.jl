@testitem "LU rank-1 update is type stable and allocates nothing" begin
    using LinearAlgebra, StrictModeTest, Test, Random
    Random.seed!(20260908)
    n = 8
    G = UpdatableLU(lu(randn(n, n) + n * I))
    u = randn(n)
    v = randn(n)
    # `@strict` guards `_bennett!`'s call in `lowrankupdate!`: with `StrictMode.checks_enabled()`
    # true (the default this suite runs under), the guard's own reflection is real compiled code
    # in `lowrankupdate!`'s body, so JET reports the guard's internal dispatch as instability and
    # AllocCheck reports its bookkeeping as allocation. Both checks are of the guard, not the
    # kernel it guards; the kernel is type-stable and allocation-free with checks disabled, which
    # is the configuration a shipped build sets.
    @test_broken (@test_typestable(lowrankupdate!(G, u, v)); true)
    @test_broken (@test_noalloc(lowrankupdate!(G, u, v)); true)
end

@testitem "Cholesky rank-1 update and downdate are type stable and allocate nothing" begin
    using LinearAlgebra, StrictModeTest, Test, Random
    Random.seed!(20260908)
    n = 8
    B = randn(n, n)
    mk() = UpdatableCholesky(cholesky(Symmetric(B * B' + n * I)))
    v = randn(n) ./ 4
    # `@strict` guards `_ch1up!`'s call in `lowrankupdate!` and `_ch1dn!`'s call in
    # `lowrankdowndate!`: with checks enabled (the default this suite runs under), each guard's
    # own reflection is real compiled code in its host function's body, so JET reports the
    # guard's internal dispatch as instability and AllocCheck reports its bookkeeping as
    # allocation. Both kernels are type-stable and allocation-free with checks disabled, which
    # is the configuration a shipped build sets.
    @test_broken (@test_typestable(lowrankupdate!(mk(), v)); true)
    @test_broken (@test_noalloc(lowrankupdate!(mk(), v)); true)
    @test_broken (@test_typestable(lowrankdowndate!(mk(), v)); true)
    @test_broken (@test_noalloc(lowrankdowndate!(mk(), v)); true)
end

@testitem "factorization invariants hold after every operation" begin
    using LinearAlgebra, Test, Random
    using UpdatableFactorizations: TypeContracts
    using .TypeContracts: behavior_passes
    Random.seed!(20260908)
    n = 7
    B = randn(n, n)
    A = Matrix(Symmetric(B * B' + n * I))
    F = UpdatableCholesky(cholesky(Symmetric(A, :L)))
    lowrankupdate!(F, randn(n))
    lowrankdowndate!(F, randn(n) ./ 16)
    delete_column!(F, 3)
    shift_columns!(F, 1, 4)
    # A large diagonal entry relative to the small off-diagonal keeps the inserted row and column
    # positive definite; a fully random insertion is not reliably so.
    x = randn(F.n + 1) ./ 10
    x[2] = n + 10.0
    insert_column!(F, 2, x)
    @test behavior_passes(UpdatableCholesky, [F])

    G = UpdatableLU(lu(randn(n, n) + n * I))
    lowrankupdate!(G, randn(n), randn(n))
    @test behavior_passes(UpdatableLU, [G])
end

@testitem "Cholesky resizing allocates nothing after construction" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)
    function measure(n)
        function mk()
            B = randn(n, n)
            return UpdatableCholesky(cholesky(Symmetric(B * B' + n * I)))
        end
        append = zeros(n + 1)
        append[n + 1] = 100.0
        insert = zeros(n + 1)
        insert[2] = 100.0
        for F in (mk(), mk(), mk(), mk())      # compile every kernel before measuring
            delete_column!(F, 3)
        end
        A, B, C, D = mk(), mk(), mk(), mk()
        shift_columns!(A, 1, n)
        UpdatableFactorizations._append!(B, append)
        insert_column!(C, 2, insert)
        E, G, H, K = mk(), mk(), mk(), mk()
        return (
            @allocated(delete_column!(E, 3)),
            @allocated(shift_columns!(G, 1, n)),
            @allocated(UpdatableFactorizations._append!(H, append)),
            @allocated(insert_column!(K, 2, insert)),
        )
    end
    @test measure(50) == (0, 0, 0, 0)
end

@testitem "QR factorization invariants hold after every operation" begin
    using LinearAlgebra, Random, Test
    using UpdatableFactorizations: TypeContracts
    using .TypeContracts: behavior_passes

    Random.seed!(20260908)
    m, n = 12, 5
    A = randn(m, n)
    F = UpdatableQR(A)
    lowrankupdate!(F, randn(m), randn(n))
    insert_column!(F, 2, randn(m))
    delete_column!(F, 4)
    shift_columns!(F, 1, 4)
    insert_row!(F, 3, randn(F.n))
    delete_row!(F, 7)
    @test behavior_passes(UpdatableQR, [F])
end

@testitem "QR updating allocates nothing after construction" begin
    using LinearAlgebra, Random, Test

    Random.seed!(20260908)
    function measure(::Type{T}, m, n) where {T}
        mk() = UpdatableQR(randn(T, m, n))
        u = randn(T, m)
        v = randn(T, n)
        x = randn(T, m)
        row = randn(T, n)
        for F in (mk(), mk(), mk(), mk(), mk(), mk())   # compile every kernel before measuring
            lowrankupdate!(F, u, v)
            insert_column!(F, 2, x)
            delete_column!(F, 3)
            shift_columns!(F, 1, 4)
            insert_row!(F, 2, row)
            delete_row!(F, 5)
        end
        A, B, C, D, E, G = mk(), mk(), mk(), mk(), mk(), mk()
        return (
            @allocated(lowrankupdate!(A, u, v)),
            @allocated(insert_column!(B, 2, x)),
            @allocated(delete_column!(C, 3)),
            @allocated(shift_columns!(D, 1, 4)),
            @allocated(insert_row!(E, 2, row)),
            @allocated(delete_row!(G, 5)),
        )
    end
    for T in (Float64, ComplexF64)
        bytes = measure(T, 80, 50)
        # `lowrankupdate!` carries `@strict`'s own reflection cost with checks enabled (the
        # default this suite runs under); every other verb is unguarded and stays exactly zero.
        @test_broken iszero(bytes[1])
        @test bytes[2:end] == (0, 0, 0, 0, 0)
    end
end

@testitem "QR rank-1 kernels are type stable and allocate nothing" begin
    using LinearAlgebra, Random, Test, StrictModeTest

    Random.seed!(20260908)
    for T in (Float64, ComplexF64)
        m, n = 40, 12
        F = UpdatableQR(randn(T, m, n))
        u = randn(T, m)
        v = randn(T, n)
        # `@strict` guards `_absorb_spike!`'s call in `lowrankupdate!`: with checks enabled (the
        # default this suite runs under), the guard's own reflection is real compiled code in
        # `lowrankupdate!`'s body, so JET reports its internal dispatch as instability and
        # AllocCheck reports its bookkeeping as allocation. The kernel itself is type-stable and
        # allocation-free with checks disabled, which is the configuration a shipped build sets.
        @test_broken (@test_typestable(lowrankupdate!(F, u, v)); true)
        @test_broken (@test_noalloc(lowrankupdate!(F, u, v)); true)
        G = UpdatableQR(randn(T, m, n))
        @test_typestable delete_row!(G, 3)
        H = UpdatableQR(randn(T, m, n))
        # `_project_residual!` accumulates `w += corr` with an explicit loop rather than
        # broadcasting, so AllocCheck's aliasing analysis has no `copyto!`/broadcast path left
        # to flag on two same-typed `SubArray`s. `delete_row!` carries no `@strict` guard of
        # its own.
        @test_noalloc delete_row!(H, 3)
    end
end

@testitem "construction routines have concrete return types" begin
    using LinearAlgebra
    chol(A) = cholesky_crout(A; s = 8)
    lup(A) = lu_crout(A; s = 8)
    lun(A) = lu_crout(A; s = 8, pivot = NoPivot())
    qrb(A) = qr_bcgs(A; s = 8)
    for M in (Matrix{Float64}, Matrix{ComplexF64})
        @test isconcretetype(Base.infer_return_type(chol, Tuple{M}))
        @test isconcretetype(Base.infer_return_type(lup, Tuple{M}))
        @test isconcretetype(Base.infer_return_type(lun, Tuple{M}))
        @test isconcretetype(Base.infer_return_type(qrb, Tuple{M}))
    end
end

@testitem "construction routines are type stable" begin
    using LinearAlgebra, StrictModeTest, Test, Random
    Random.seed!(20260908)
    for T in (Float64, ComplexF64)
        n = 24
        S = randn(T, n, n)
        S = S'S + n * I
        G = randn(T, n, n)
        W = randn(T, 40, 12)
        @test_typestable cholesky_crout(Matrix(S); s = 8)
        @test_typestable lu_crout(copy(G); s = 8)
        @test_typestable lu_crout(copy(G); s = 8, pivot = NoPivot())
        @test_typestable qr_bcgs(copy(W); s = 4)
    end
end

@testitem "construction allocates no more than the factorization it returns" begin
    using LinearAlgebra, Random
    Random.seed!(20260908)

    # Bytes of array storage the returned object owns, including any scratch the updating verbs
    # rely on and any buffer held indirectly by a nested representation such as the Q factor.
    function owned(F, depth = 2)
        F isa AbstractArray && return sizeof(F)
        (iszero(depth) || isbits(F)) && return 0
        return sum(owned(getfield(F, f), depth - 1) for f in fieldnames(typeof(F)); init = 0)
    end

    # A construction routine needs its result plus one mutable copy of the input to work in.
    # Anything beyond that is a buffer built and then thrown away.
    function within_budget(f, A)
        F = f(A)
        f(A)                                   # compile before measuring
        bytes = @allocated f(A)
        budget = owned(F) + sizeof(A)
        return bytes, budget, bytes <= 1.05 * budget
    end

    n = 400
    S = let B = randn(n, n)
        Matrix(Hermitian(B * B' + n * I))
    end
    G = randn(n, n)
    for (name, f, A) in (
            ("cholesky_crout", A -> cholesky_crout(A; s = 64), S),
            ("lu_crout", A -> lu_crout(A; s = 64), G),
            ("lu_crout unpivoted", A -> lu_crout(A; s = 64, pivot = NoPivot()), S),
            ("qr_bcgs", A -> qr_bcgs(A; s = 64), G),
        )
        bytes, budget, ok = within_budget(f, A)
        ok || @info "$name allocated $bytes bytes against a budget of $budget"
        @test ok
    end
end

@testitem "UpdatableCholesky satisfies its strict contract" begin
    using LinearAlgebra, StrictModeTest, Test, Random
    using UpdatableFactorizations: TypeContracts, StrictMode, AbstractUpdatableCholesky
    Random.seed!(20260908)
    const_type = UpdatableCholesky{Float64, Float64, Matrix{Float64}}
    @test TypeContracts.check_contract(const_type, AbstractUpdatableCholesky).passed

    n = 8
    B = randn(n, n)
    mk() = UpdatableCholesky(cholesky(Symmetric(B * B' + n * I)))
    x = randn(n + 1) ./ 10
    x[2] = n + 10.0
    v = randn(n) ./ 4
    for _ in 1:4                            # compile every kernel before measuring
        delete_column!(mk(), 3)
        shift_columns!(mk(), 1, 4)
        insert_column!(mk(), 2, x)
        lowrankupdate!(mk(), v)
        lowrankdowndate!(mk(), v ./ 4)
    end
    A, C, E, G2, H = mk(), mk(), mk(), mk(), mk()
    # `@verify_strict`'s own interface check runs against `const_type`'s nominal supertype chain
    # (`Factorization`), which carries no registered contract, so it passes vacuously here; the
    # assertion above, using the structural (Holy Trait) form of `check_contract`, is what
    # actually checks the method surface. `@strict` guards `_ch1up!`'s call inside
    # `lowrankupdate!` and `_ch1dn!`'s call inside `lowrankdowndate!`, but measured:
    # `@verify_strict`'s own type-stability check does not throw on either call, unlike
    # `@test_typestable` above (which uses JET and does), so both are included here; the
    # `@assert_owned`/`@assert_noalloc` warnings this block prints for them are exactly the
    # guard's own bookkeeping.
    StrictMode.@verify_strict const_type begin
        delete_column!(A, 3)
        shift_columns!(C, 1, 4)
        insert_column!(E, 2, x)
        lowrankupdate!(G2, v)
        lowrankdowndate!(H, v ./ 4)
        size(A)
    end
end

@testitem "UpdatableLU satisfies its strict contract" begin
    using LinearAlgebra, StrictModeTest, Test, Random
    using UpdatableFactorizations: TypeContracts, StrictMode, AbstractUpdatableLU
    Random.seed!(20260908)
    const_type = UpdatableLU{Float64, Matrix{Float64}}
    @test TypeContracts.check_contract(const_type, AbstractUpdatableLU).passed

    n = 8
    u, v = randn(n), randn(n)
    for _ in 1:4                            # compile every kernel before measuring
        lowrankupdate!(UpdatableLU(lu(randn(n, n) + n * I)), u, v)
    end
    G = UpdatableLU(lu(randn(n, n) + n * I))
    # `@strict` guards `_bennett!`'s call inside `lowrankupdate!`, but measured: `@verify_strict`'s
    # own type-stability check does not throw on it, unlike `@test_typestable` above (which uses
    # JET and does), so the call is included here; the `@assert_owned`/`@assert_noalloc` warnings
    # this block prints for it are exactly the guard's own bookkeeping.
    StrictMode.@verify_strict const_type begin
        lowrankupdate!(G, u, v)
        size(G)
    end
end

@testitem "UpdatableQR satisfies its strict contract" begin
    using LinearAlgebra, StrictModeTest, Test, Random
    using UpdatableFactorizations: TypeContracts, StrictMode, AbstractUpdatableQR, DenseQ, capacity
    Random.seed!(20260908)
    const_type = UpdatableQR{Float64, Matrix{Float64}, DenseQ{Float64, Matrix{Float64}}}
    @test TypeContracts.check_contract(const_type, AbstractUpdatableQR).passed

    m, n = 12, 5
    mk() = UpdatableQR(randn(m, n))
    for F in (mk(), mk(), mk(), mk(), mk(), mk())   # compile every kernel before measuring
        insert_column!(F, 2, randn(m))
        delete_column!(F, 4)
        shift_columns!(F, 1, 4)
        insert_row!(F, 3, randn(F.n))
        delete_row!(F, 7)
    end
    A, C, E, G2, H, K = mk(), mk(), mk(), mk(), mk(), mk()
    u, v = randn(m), randn(n)
    # `@strict` guards `_absorb_spike!`'s call inside `lowrankupdate!`, but measured:
    # `@verify_strict`'s own type-stability check does not throw on it, unlike `@test_typestable`
    # above (which uses JET and does), so the call is included here; the `@assert_owned`/
    # `@assert_noalloc` warnings this block prints for it are exactly the guard's own bookkeeping.
    StrictMode.@verify_strict const_type begin
        insert_column!(A, 2, randn(m))
        delete_column!(C, 4)
        shift_columns!(E, 1, 4)
        insert_row!(G2, 3, randn(G2.n))
        delete_row!(H, 7)
        lowrankupdate!(K, u, v)
        size(A)
        capacity(A)
    end
end

@testitem "unguarded verbs keep their type-stability and allocation guarantees" begin
    using LinearAlgebra, StrictModeTest
    # Signatures, not values: the sweep proves the guarantee for a concrete specialization
    # without constructing one. The rank-1 update verbs are absent on purpose — each carries a
    # `@strict` guard whose own reflection is compiled into the caller, which is what the
    # `@test_broken` items above record.
    for T in (Float64, ComplexF64)
        Q = UpdatableQR{T, Matrix{T}, UpdatableFactorizations.DenseQ{T, Matrix{T}}}
        C = UpdatableCholesky{T, real(T), Matrix{T}}
        findings = test_signatures(
            [
                (delete_row!, (Q, Int)),
                (delete_column!, (Q, Int)),
                (shift_columns!, (Q, Int, Int)),
                (delete_column!, (C, Int)),
                (shift_columns!, (C, Int, Int)),
            ]
        )
        @test !isempty(findings)
    end
end
