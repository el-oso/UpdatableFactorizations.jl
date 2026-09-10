using Test, Random
include(joinpath(@__DIR__, "harness.jl"))

# Every cell function below builds its matrices with randn, and the checks assert a residual
# bound on the result. The seed makes that bound reproducible.
Random.seed!(20260908)

@testset "harness" begin
    @test median([3.0, 1.0, 2.0]) == 2.0
    @test median([4.0, 1.0, 3.0, 2.0]) == 2.0

    single_threaded!()
    @test BLAS.get_num_threads() == 1
    interfaces = [c.interface for c in BLAS.lbt_get_config().loaded_libs]
    @test :lp64 in interfaces      # libqrupdate resolves its BLAS symbols through this one
    @test :ilp64 in interfaces     # Julia's own linear algebra keeps the library it had

    @test governor() isa String

    # `@mutating_bench`/`@pure_bench` reject a stray positional argument -- the shape `@be`
    # would otherwise resolve as `teardown` and silently not time -- and always run with
    # `evals = 1`.
    mkv() = [1]
    b = @mutating_bench(mkv, v -> push!(v, 2), samples = 5)
    @test length(b.samples) == 5
    b2 = @pure_bench(sum(collect(1:8)), samples = 5)
    @test length(b2.samples) == 5
    @test_throws ErrorException Base.macroexpand(
        Main, :(@mutating_bench(mkv, f, teardown))
    )
    @test_throws ErrorException Base.macroexpand(Main, :(@pure_bench(sum([1, 2]), identity)))

    empty!(ROWS)
    record!(;
        family = "probe", routine = "sum", variant = "Base", eltype = Float64, n = 8,
        bench = @be(zeros(8), sum, evals = 1, samples = 5), relerr = 0.0,
    )
    @test length(ROWS) == 1
    @test ROWS[1]["family"] == "probe"
    @test length(ROWS[1]["samples"]) == 5
    @test ROWS[1]["median_seconds"] > 0

    path = save_results("probe")
    meta = JSON.parsefile(path)
    @test meta["host"] == gethostname()
    @test length(meta["rows"]) == 1
    @test meta["rows"][1]["routine"] == "sum"
    rm(path)
    empty!(ROWS)
end

@testset "cholesky cells" begin
    include(joinpath(@__DIR__, "updating_sweep.jl"))
    empty!(ROWS)
    cholesky_cells((24,); samples = 5)
    routines = unique(r["routine"] for r in ROWS)
    @test issetequal(
        routines,
        [
            "lowrankupdate!", "lowrankdowndate!", "delete_column!",
            "append_column!", "insert_column!", "shift_columns!",
        ]
    )
    @test all(r -> r["family"] == "cholesky", ROWS)
    @test all(r -> r["relerr"] < 1.0e-12, ROWS)
    @test any(r -> r["variant"] == "LinearAlgebra", ROWS)
    @test any(r -> r["variant"] == "UpdatableCholeskyFactorizations", ROWS)
    @test any(r -> r["variant"] == "recompute", ROWS)
    @test all(
        r -> iszero(r["median_bytes"]),
        filter(r -> r["variant"] == "UpdatableFactorizations", ROWS)
    )
    empty!(ROWS)
end

@testset "lu cells" begin
    empty!(ROWS)
    lu_cells((24,); samples = 5)
    @test issetequal(
        unique(r["routine"] for r in ROWS),
        ["lowrankupdate! nopivot", "lowrankupdate! pivoted"]
    )
    @test all(r -> r["family"] == "lu", ROWS)
    # QRupdatesFast's pivoted routine reaches libqrupdate's BLAS calls. Without the LP64
    # forwarding in single_threaded! it returns a residual near one instead of near zero.
    @test all(r -> r["relerr"] < 1.0e-12, ROWS)
    @test any(r -> r["variant"] == "QRupdatesFast", ROWS)
    @test all(
        r -> iszero(r["median_bytes"]),
        filter(r -> r["variant"] == "UpdatableFactorizations", ROWS)
    )
    empty!(ROWS)
end

# UpdatableFactorizations ships only DenseQ; there is no second Q representation to benchmark
# alongside it, so every QR row below compares DenseQ against the prior-art packages and
# recomputing from scratch.
@testset "qr cells" begin
    empty!(ROWS)
    qr_cells(((48, 12),); samples = 5)
    @test issetequal(
        unique(r["routine"] for r in ROWS),
        [
            "lowrankupdate!", "insert_column!", "delete_column!", "shift_columns!",
            "insert_row!", "delete_row!",
        ]
    )
    @test all(r -> r["family"] == "qr", ROWS)
    @test all(r -> r["m"] == 48 && r["n"] == 12, ROWS)
    @test all(r -> r["relerr"] < 1.0e-12, ROWS)
    # One row per cell in the case matrix: 2 + 4 + 4 + 3 + 3 + 2 for a single size. A dropped
    # cell changes this count, which the routine-set assertion above cannot see.
    @test length(ROWS) == 18
    # Each prior-art package is present. `all` over an empty generator is true, so the flag
    # assertions below say nothing unless the rows they filter for actually exist.
    @test any(r -> r["variant"] == "QRupdate", ROWS)
    @test any(r -> r["variant"] == "UpdatableQRFactorizations", ROWS)
    @test any(r -> r["variant"] == "QRupdatesFast", ROWS)
    @test any(r -> r["variant"] == "recompute", ROWS)
    # The two comparisons that do a different amount of work are flagged, not silently ranked.
    @test all(get(r, "r_only", false) for r in ROWS if r["variant"] == "QRupdate")
    @test all(
        get(r, "full_q", false) for r in ROWS if r["variant"] == "UpdatableQRFactorizations"
    )
    @test all(
        r -> iszero(r["median_bytes"]),
        filter(r -> r["variant"] == "UpdatableFactorizations", ROWS)
    )
    empty!(ROWS)
end

@testset "plot_results reads only the saved data" begin
    src = read(joinpath(@__DIR__, "plot_results.jl"), String)
    # The script must not construct a factorization or call a benchmark macro: the published
    # numbers come from the recorded file, not from a fresh measurement.
    for forbidden in ("@be", "@b ", "UpdatableCholesky(", "UpdatableLU(", "UpdatableQR(")
        @test !occursin(forbidden, src)
    end
    @test occursin("JSON.parsefile", src)
end
