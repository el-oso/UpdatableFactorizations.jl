# Whole-loop cost of an active-set working-set change: build the factorization, then alternate
# column insertions and deletions with solves. The representations differ in where the cost of
# the orthonormal factor falls -- formed once and applied cheaply, or never formed and applied
# through a growing rotation log -- so only the total over a run decides between them.
#
# `ratio` here is `dense_time / givens_time`: above 1.00 means GivensQ is the faster arm. This is
# the opposite convention from `construct_sweep.jl`, whose ratio is candidate over baseline with
# the candidate in the numerator; every number printed or written below states its direction.
using LinearAlgebra, Random, JSON, Printf, Dates
using UpdatableFactorizations

include(joinpath(@__DIR__, "..", "docs", "superpowers", "specs", "proto_givensq.jl"))

BLAS.set_num_threads(1)

function dense_loop(A, xs, bs)
    m, n = size(A)
    F = UpdatableQR(A; capacity = (m, n + length(xs)))
    s = 0.0
    for k in eachindex(xs)
        insert_column!(F, size(F, 2) + 1, xs[k])
        s += sum(F \ bs[k])
        delete_column!(F, 1)
        s += sum(F \ bs[k])
    end
    return s
end

function givens_loop(A, xs, bs)
    m = size(A, 1)
    G = qr(A)
    Q = GQ(G)
    R = Matrix(G.R)
    w = zeros(eltype(A), m)
    y = zeros(eltype(A), m)
    s = 0.0
    for k in eachindex(xs)
        # Insertion: the projection, the new direction, and the widened triangle.
        applyadj!(view(w, 1:Q.n), Q, xs[k])
        apply!(y, Q, view(w, 1:Q.n))
        r = xs[k] - y
        beta = augment!(Q, r)
        R = [R view(w, 1:(Q.n - 1)); zeros(1, Q.n - 1) beta]
        s += sum(UpperTriangular(R) \ (applyadj!(zeros(Q.n), Q, bs[k])))
        # Deletion of the leading column: drop it and retriangularize, pushing each rotation.
        R = R[:, 2:end]
        for i in 1:(size(R, 2))
            H, _ = LinearAlgebra.givens(R[i, i], R[i + 1, i], i, i + 1)
            lmul!(H, R)
            push_rot!(Q, Givens(H.i1, H.i2, real(H.c), -H.s))
        end
        R = R[1:(end - 1), :]
        Q.n -= 1
        s += sum(UpperTriangular(R) \ (applyadj!(zeros(Q.n), Q, bs[k])))
    end
    return s
end

const ROUNDS = 20

# One comparison: alternate single calls of the dense and GivensQ loops, `ROUNDS` times, after
# one untimed warmup call to each that pays for compilation. `GC.gc()` runs once per round,
# before either call, so a GC pause tripped by one side's allocation does not land unevenly
# across rounds; see `bench/construct_sweep.jl`'s `paired`, which this mirrors. `@timed` gives
# both a wall time and the GC time within it, so both a wall ratio (what a caller feels) and a
# gc-net ratio (wall minus GC time, which is not at the mercy of which side a pause lands on) are
# recorded per round.
function paired(dense, givens; rounds = ROUNDS)
    sd = dense()
    sg = givens()
    wall_d = Float64[]
    wall_g = Float64[]
    gcnet_d = Float64[]
    gcnet_g = Float64[]
    for _ in 1:rounds
        GC.gc()
        rd = @timed (sd = dense())
        push!(wall_d, rd.time)
        push!(gcnet_d, rd.time - rd.gctime)
        rg = @timed (sg = givens())
        push!(wall_g, rg.time)
        push!(gcnet_g, rg.time - rg.gctime)
    end
    return wall_d, wall_g, gcnet_d, gcnet_g, sd, sg
end

median(v) = sort(v)[(length(v) + 1) ÷ 2]

function cell(m, n, k; rounds = ROUNDS)
    Random.seed!(20260908)
    A = randn(m, n)
    xs = [randn(m) for _ in 1:k]
    bs = [randn(m) for _ in 1:k]
    dense = () -> dense_loop(A, xs, bs)
    givens = () -> givens_loop(A, xs, bs)
    wall_d, wall_g, gcnet_d, gcnet_g, sd, sg = paired(dense, givens; rounds)
    ratio_wall = wall_d ./ wall_g
    ratio_gcnet = gcnet_d ./ gcnet_g
    return Dict{String, Any}(
        "m" => m, "n" => n, "k" => k,
        "dense_wall_seconds" => wall_d, "givens_wall_seconds" => wall_g,
        "dense_gcnet_seconds" => gcnet_d, "givens_gcnet_seconds" => gcnet_g,
        "dense_wall_median_seconds" => median(wall_d),
        "givens_wall_median_seconds" => median(wall_g),
        "dense_gcnet_median_seconds" => median(gcnet_d),
        "givens_gcnet_median_seconds" => median(gcnet_g),
        # ratio = dense_time / givens_time; above 1.00 means GivensQ is faster.
        "ratio_wall_samples" => ratio_wall, "ratio_wall_median" => median(ratio_wall),
        "ratio_wall_min" => minimum(ratio_wall), "ratio_wall_max" => maximum(ratio_wall),
        "ratio_gcnet_samples" => ratio_gcnet, "ratio_gcnet_median" => median(ratio_gcnet),
        "ratio_gcnet_min" => minimum(ratio_gcnet), "ratio_gcnet_max" => maximum(ratio_gcnet),
        "relerr" => abs(sd - sg) / abs(sd),
    )
end

# Shapes an active-set QP solver reaches: many more rows than working-set columns, and a
# working set that turns over on the order of once per column.
shapes = ((400, 50), (800, 100), (2000, 200))
rows = [cell(m, n, k) for (m, n) in shapes for k in (n ÷ 4, n ÷ 2, n)]

mkpath(joinpath(@__DIR__, "results"))
open(joinpath(@__DIR__, "results", "givensq_decision.json"), "w") do io
    JSON.print(
        io,
        Dict{String, Any}(
            "julia" => string(VERSION), "date" => string(Dates.today()),
            "rounds" => ROUNDS,
            "ratio_convention" => "dense_time / givens_time; above 1.00 means GivensQ is faster",
            "rows" => rows,
        )
    )
end

println("ratio = dense_time / givens_time; above 1.00 means GivensQ is faster")
for r in rows
    @printf(
        "m=%-5d n=%-5d k=%-5d  wall %6.2fx [%.2f, %.2f]  gc-net %6.2fx [%.2f, %.2f]  relerr %.1e\n",
        r["m"], r["n"], r["k"],
        r["ratio_wall_median"], r["ratio_wall_min"], r["ratio_wall_max"],
        r["ratio_gcnet_median"], r["ratio_gcnet_min"], r["ratio_gcnet_max"], r["relerr"]
    )
end
