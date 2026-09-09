# Sweep of the construction layer against the standard library. Writes every sample, not just
# the median, so the plots and tables can be regenerated without re-running.
using UpdatableFactorizations
using LinearAlgebra, ForwardDiff, Chairmarks, JSON, Random, Printf, Dates

BLAS.set_num_threads(1)
Random.seed!(20260908)

const ROWS = Dict{String, Any}[]

function record(;
        routine, variant, eltype, n, s, bench, relerr::Float64,
        extra = Dict{String, Any}()
    )
    times = [smp.time for smp in bench.samples]
    row = Dict{String, Any}(
        "routine" => routine, "variant" => variant, "eltype" => string(eltype),
        "n" => n, "s" => s, "median_seconds" => median(times), "samples" => times,
        "relerr" => relerr,
    )
    merge!(row, extra)
    push!(ROWS, row)
    @printf(
        "%-10s %-16s %-8s n=%-5d s=%-5s %8.1f ms  relerr %.1e\n",
        routine, variant, string(eltype), n, isnothing(s) ? "-" : string(s),
        1.0e3 * median(times), relerr
    )
    return row
end

spd(T, n) = (B = randn(T, n, n); Matrix(Hermitian(B * B' + n * I)))
median(v) = sort(v)[(length(v) + 1) ÷ 2]

# `norm` over a Dual matrix returns a Dual, which has no Float64 conversion; the value is the
# part the sweep records.
scalar(x) = Float64(x)
scalar(x::ForwardDiff.Dual) = Float64(ForwardDiff.value(x))

function cholesky_cells(ns, ss)
    for n in ns
        A = spd(Float64, n)
        record(;
            routine = "cholesky", variant = "lapack potrf", eltype = Float64, n, s = nothing,
            bench = @be(copy(A), cholesky!(Symmetric(_, :L)), evals = 1, samples = 200),
            relerr = norm(Matrix(cholesky(Symmetric(A, :L))) - A) / norm(A)
        )
        gemm_rankk!(C, X, alpha, beta; uplo = 'L') = mul!(C, X, X', alpha, beta)
        for s in ss,
                (name, f) in (
                    ("rankk", UpdatableFactorizations.default_rankk!), ("gemm", gemm_rankk!),
                )
            s < n || continue
            F = cholesky_crout(A; s, rankk! = f)
            record(;
                routine = "cholesky", variant = name, eltype = Float64, n, s,
                bench = @be(cholesky_crout(A; s, rankk! = f), evals = 1, samples = 200),
                relerr = norm(Matrix(F) - A) / norm(A)
            )
        end
    end
    return
end

function lu_cells(ns, ss)
    dominant = Dict{String, Any}("matrix" => "dominant")
    random = Dict{String, Any}("matrix" => "random")
    for n in ns
        # Two fixtures. The diagonally dominant one has nonzero leading minors, which is what the
        # unpivoted factorization needs to be accurate, but partial pivoting picks the diagonal
        # at every column and so never swaps a row. The plain random one pivots, and it is the
        # only fixture on which the cost of pivoting is visible.
        Ad = Matrix(randn(n, n) + n * I)
        Ar = randn(n, n)
        record(;
            routine = "lu", variant = "lapack getrf", eltype = Float64, n, s = nothing,
            bench = @be(copy(Ar), lu!(_), evals = 1, samples = 200),
            relerr = norm(Matrix(lu(Ar)) - Ar) / norm(Ar), extra = random
        )
        record(;
            routine = "lu", variant = "stdlib nopivot", eltype = Float64, n, s = nothing,
            bench = @be(copy(Ad), lu!(_, NoPivot()), evals = 1, samples = 200),
            relerr = norm(Matrix(lu(Ad, NoPivot())) - Ad) / norm(Ad), extra = dominant
        )
        for s in ss
            s < n || continue
            F = lu_crout(Ad; s, pivot = NoPivot())
            record(;
                routine = "lu", variant = "crout nopivot", eltype = Float64, n, s,
                bench = @be(lu_crout(Ad; s, pivot = NoPivot()), evals = 1, samples = 200),
                relerr = norm(F.L * F.U - Ad) / norm(Ad), extra = dominant
            )
            # The pivoted and unpivoted timings on the same pivoting matrix. The unpivoted
            # residual here is not a quality number: without pivoting a random matrix has no
            # stability bound, and only the time is being compared.
            H = lu_crout(Ar; s, pivot = NoPivot())
            record(;
                routine = "lu", variant = "crout nopivot", eltype = Float64, n, s,
                bench = @be(lu_crout(Ar; s, pivot = NoPivot()), evals = 1, samples = 200),
                relerr = norm(H.L * H.U - Ar) / norm(Ar), extra = random
            )
            G = lu_crout(Ar; s)
            record(;
                routine = "lu", variant = "crout rowmaximum", eltype = Float64, n, s,
                bench = @be(lu_crout(Ar; s), evals = 1, samples = 200),
                relerr = norm(G.L * G.U - Ar[G.p, :]) / norm(Ar),
                extra = merge(random, Dict{String, Any}("interchanges" => count(G.p .!= 1:n)))
            )
        end
    end
    return
end

function qr_cells(ns, ss)
    for n in ns
        A = randn(n, n)
        F = qr(A); Qh = Matrix(F.Q)
        record(;
            routine = "qr", variant = "lapack geqrf", eltype = Float64, n, s = nothing,
            bench = @be(copy(A), qr!(_), evals = 1, samples = 200), relerr = norm(Qh * F.R - A) / norm(A),
            extra = Dict{String, Any}("ortherr" => norm(Qh' * Qh - I))
        )
        record(;
            routine = "qr", variant = "lapack geqrf+Q", eltype = Float64, n, s = nothing,
            bench = @be(copy(A), (G = qr!(_); Matrix(G.Q)), evals = 1, samples = 200),
            relerr = norm(Qh * F.R - A) / norm(A),
            extra = Dict{String, Any}("ortherr" => norm(Qh' * Qh - I))
        )
        for s in ss, reorth in (false, true)
            s < n || continue
            Fq = qr_bcgs(A; s, reorth)
            record(;
                routine = "qr", variant = "bcgs reorth=$reorth", eltype = Float64, n, s,
                bench = @be(qr_bcgs(A; s, reorth), evals = 1, samples = 200),
                relerr = norm(Matrix(Fq) - A) / norm(A),
                extra = Dict{String, Any}("ortherr" => norm(Fq.Q' * Fq.Q - I))
            )
        end
    end
    return
end

# Does blocking buy anything when mul! is itself a scalar triple loop? Compared against the same
# routine with s >= n, which never flushes and so is the unblocked left-looking factorization.
function nonblas_cells()
    for (T, n) in ((BigFloat, 120), (ForwardDiff.Dual{Nothing, Float64, 1}, 200))
        A = T.(spd(Float64, n))
        for s in (16, 64, n)
            F = cholesky_crout(A; s)
            record(;
                routine = "cholesky", variant = s == n ? "unblocked" : "rankk",
                eltype = T, n, s, bench = @be(cholesky_crout(A; s), evals = 1, samples = 200, seconds = 30),
                relerr = scalar(norm(Matrix(F) - A) / norm(A))
            )
        end
        record(;
            routine = "cholesky", variant = "stdlib generic", eltype = T, n, s = nothing,
            bench = @be(cholesky(Hermitian(A, :L)), evals = 1, samples = 200, seconds = 30),
            relerr = scalar(norm(Matrix(cholesky(Hermitian(A, :L))) - A) / norm(A))
        )
    end
    return
end

cholesky_cells((2000, 4000), (64, 128, 256))
lu_cells((2000, 4000), (64, 128, 256))
qr_cells((2000, 4000), (32, 64, 128, 256))
nonblas_cells()

meta = Dict{String, Any}(
    "host" => gethostname(), "julia" => string(VERSION), "blas" => BLAS.get_config() |> string,
    "blas_threads" => BLAS.get_num_threads(), "date" => string(Dates.today()),
    "rows" => ROWS,
)
mkpath(joinpath(@__DIR__, "results"))
open(joinpath(@__DIR__, "results", "construction-$(gethostname()).json"), "w") do io
    JSON.print(io, meta)
end
