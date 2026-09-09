# Sweep of the construction layer against the standard library. Every comparison alternates
# single calls of its two sides in one loop, so a clock or thermal drift over the run moves both
# sides together and the per-round ratio cancels it; a bare ratio of two independently measured
# medians would not. Every round is written out, not just the median, so tables and plots are
# regenerated from the saved JSON rather than by re-running.
using UpdatableFactorizations
using LinearAlgebra, ForwardDiff, JSON, Random, Printf, Dates

BLAS.set_num_threads(1)
Random.seed!(20260908)

const ROWS = Dict{String, Any}[]
const ROUNDS = 12

# One comparison: alternate single calls of `base` and `cand`. The first call to each compiles
# and primes the allocator and is not timed; every later call is a genuine sample. Returns the
# per-round times for both sides plus the value each produced on its last call, for correctness
# checks that would otherwise cost one more expensive call at the largest sizes.
function paired(base, cand; rounds = ROUNDS)
    fb = base()
    fc = cand()
    tb = Float64[]
    tc = Float64[]
    for _ in 1:rounds
        push!(tb, @elapsed (fb = base()))
        push!(tc, @elapsed (fc = cand()))
    end
    return tb, tc, fb, fc
end

median(v) = sort(v)[(length(v) + 1) ÷ 2]

function record(;
        routine, baseline, variant, eltype, n, s, tb, tc, relerr::Float64,
        extra = Dict{String, Any}()
    )
    ratio = tc ./ tb
    row = Dict{String, Any}(
        "routine" => routine, "baseline" => baseline, "variant" => variant,
        "eltype" => string(eltype), "n" => n, "s" => s,
        "base_samples" => tb, "cand_samples" => tc,
        "base_median_seconds" => median(tb), "cand_median_seconds" => median(tc),
        "ratio_samples" => ratio, "ratio_median" => median(ratio),
        "ratio_min" => minimum(ratio), "ratio_max" => maximum(ratio),
        "relerr" => relerr,
    )
    merge!(row, extra)
    push!(ROWS, row)
    @printf(
        "%-10s %-18s vs %-16s %-8s n=%-5d s=%-5s ratio %6.2fx [%.2f, %.2f]  relerr %.1e\n",
        routine, variant, baseline, string(eltype), n, isnothing(s) ? "-" : string(s),
        median(ratio), minimum(ratio), maximum(ratio), relerr
    )
    return row
end

spd(T, n) = (B = randn(T, n, n); Matrix(Hermitian(B * B' + n * I)))

# `norm` over a Dual matrix returns a Dual, which has no Float64 conversion; the value is the
# part the sweep records.
scalar(x) = Float64(x)
scalar(x::ForwardDiff.Dual) = Float64(ForwardDiff.value(x))

function cholesky_cells(ns, ss)
    gemm_rankk!(C, X, alpha, beta; uplo = 'L') = mul!(C, X, X', alpha, beta)
    for n in ns
        A = spd(Float64, n)
        base = () -> cholesky!(Symmetric(copy(A), :L))
        for s in ss,
                (name, f) in (
                    ("rankk", UpdatableFactorizations.default_rankk!), ("gemm", gemm_rankk!),
                )
            s < n || continue
            cand = () -> cholesky_crout(A; s, rankk! = f)
            tb, tc, _, Fc = paired(base, cand)
            record(;
                routine = "cholesky", baseline = "lapack potrf", variant = name,
                eltype = Float64, n, s, tb, tc, relerr = norm(Matrix(Fc) - A) / norm(A)
            )
        end
    end
    return
end

function lu_cells(ns, ss)
    dominant = Dict{String, Any}("matrix" => "dominant")
    random = Dict{String, Any}("matrix" => "random")
    for n in ns
        # Two fixtures, built once per n and reused across every s. The diagonally dominant one
        # has nonzero leading minors, which is what the unpivoted factorization needs to be
        # accurate, but partial pivoting picks the diagonal at every column and so never swaps a
        # row. The plain random one pivots, and it is the only fixture on which the cost of
        # pivoting is visible.
        Ad = Matrix(randn(n, n) + n * I)
        Ar = randn(n, n)
        getrf = () -> lu!(copy(Ar))
        stdnopivot = () -> lu!(copy(Ad), NoPivot())
        for s in ss
            s < n || continue
            row = () -> lu_crout(Ar; s)
            nopivot_ad = () -> lu_crout(Ad; s, pivot = NoPivot())
            nopivot_ar = () -> lu_crout(Ar; s, pivot = NoPivot())

            tb, tc, _, G = paired(getrf, row)
            record(;
                routine = "lu", baseline = "lapack getrf", variant = "crout rowmaximum",
                eltype = Float64, n, s, tb, tc,
                relerr = norm(G.L * G.U - Ar[G.p, :]) / norm(Ar),
                extra = merge(random, Dict{String, Any}("interchanges" => count(G.p .!= 1:n)))
            )

            tb, tc, _, F = paired(stdnopivot, nopivot_ad)
            record(;
                routine = "lu", baseline = "stdlib nopivot", variant = "crout nopivot",
                eltype = Float64, n, s, tb, tc,
                relerr = norm(F.L * F.U - Ad) / norm(Ad), extra = dominant
            )

            # The pivoted and unpivoted forms of the same algorithm on the same matrix, so the
            # ratio isolates the cost of the pivot search and row swaps with no LAPACK call on
            # either side. The residual is not a quality number: without pivoting, a random
            # matrix has no stability bound, and only the time is being compared.
            tb, tc, _, H = paired(row, nopivot_ar)
            record(;
                routine = "lu", baseline = "crout rowmaximum", variant = "crout nopivot",
                eltype = Float64, n, s, tb, tc,
                relerr = norm(H.L * H.U - Ar) / norm(Ar), extra = random
            )
        end
    end
    return
end

function qr_cells(ns, ss)
    for n in ns
        A = randn(n, n)
        geqrf = () -> qr!(copy(A))
        geqrfQ = () -> (G = qr!(copy(A)); Matrix(G.Q))

        tb, tc, Fg, Qh = paired(geqrf, geqrfQ)
        record(;
            routine = "qr", baseline = "lapack geqrf", variant = "lapack geqrf+Q",
            eltype = Float64, n, s = nothing, tb, tc,
            relerr = norm(Qh * Fg.R - A) / norm(A),
            extra = Dict{String, Any}("ortherr" => norm(Qh' * Qh - I))
        )

        for s in ss, reorth in (false, true)
            s < n || continue
            bcgs = () -> qr_bcgs(A; s, reorth)
            tb, tc, _, Fq = paired(geqrfQ, bcgs)
            record(;
                routine = "qr", baseline = "lapack geqrf+Q", variant = "bcgs reorth=$reorth",
                eltype = Float64, n, s, tb, tc,
                relerr = norm(Matrix(Fq) - A) / norm(A),
                extra = Dict{String, Any}("ortherr" => norm(Fq.Q' * Fq.Q - I))
            )
        end
    end
    return
end

# Does blocking buy anything when mul! is itself a scalar triple loop? `unblocked` (s = n) never
# flushes, so it is the left-looking factorization with no blocking at all; `stdlib generic` is
# LinearAlgebra's own fallback for a non-BLAS element type.
function nonblas_cells()
    for (T, n) in ((BigFloat, 120), (ForwardDiff.Dual{Nothing, Float64, 1}, 200))
        A = T.(spd(Float64, n))
        unblocked = () -> cholesky_crout(A; s = n)
        stdlib_generic = () -> cholesky(Hermitian(A, :L))

        tb, tc, _, Fs = paired(unblocked, stdlib_generic)
        record(;
            routine = "cholesky", baseline = "unblocked", variant = "stdlib generic",
            eltype = T, n, s = n, tb, tc, relerr = scalar(norm(Matrix(Fs) - A) / norm(A))
        )

        for s in (16, 64)
            blocked = () -> cholesky_crout(A; s)
            tb, tc, _, Fb = paired(unblocked, blocked)
            record(;
                routine = "cholesky", baseline = "unblocked", variant = "rankk",
                eltype = T, n, s, tb, tc, relerr = scalar(norm(Matrix(Fb) - A) / norm(A))
            )
        end
    end
    return
end

cholesky_cells((2000, 4000), (64, 128, 256))
lu_cells((2000, 4000), (64, 128, 256))
qr_cells((2000, 4000), (32, 64, 128, 256))
nonblas_cells()

meta = Dict{String, Any}(
    "host" => gethostname(), "julia" => string(VERSION), "blas" => BLAS.get_config() |> string,
    "blas_threads" => BLAS.get_num_threads(), "date" => string(Dates.today()), "rounds" => ROUNDS,
    "rows" => ROWS,
)
mkpath(joinpath(@__DIR__, "results"))
open(joinpath(@__DIR__, "results", "construction-$(gethostname()).json"), "w") do io
    JSON.print(io, meta)
end
