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
