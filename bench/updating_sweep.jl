# Every updating verb against the standard library, the prior-art packages and recomputing the
# factorization from scratch. Recomputing is the baseline that says whether updating pays.
#
# StrictMode's allocation and type-stability guards cost real allocation and dynamic dispatch
# when enabled, which is the state ordinary development and testing run in.
# bench/LocalPreferences.toml disables them for this environment, so every
# UpdatableFactorizations cell below measures the bare kernel, matching a shipped build, rather
# than the guards' own reflection cost.

using LinearAlgebra, Random, Chairmarks
using UpdatableFactorizations
import UpdatableCholeskyFactorizations
import UpdatableQRFactorizations
import QRupdate
import QRupdatesFast
import StrictMode

StrictMode.checks_enabled() && error(
    "StrictMode checks are enabled in this environment; every UpdatableFactorizations cell " *
        "would measure the guards' reflection cost instead of the kernel. Check " *
        "bench/LocalPreferences.toml."
)

# The harness defines `const ROWS`; including it twice into the same module would redefine that
# constant and rebind the array a caller is already collecting into.
isdefined(Main, :ROWS) || include(joinpath(@__DIR__, "harness.jl"))

# The comparison packages are reached through `import`, so their exports never enter this file's
# scope. Naming the module at every call site keeps it visible which package a verb belongs to,
# and keeps the file correct if a `using` is ever added: UpdatableCholeskyFactorizations and
# UpdatableQRFactorizations both export the names UpdatableCholesky and UpdatableQR.
const UF = UpdatableFactorizations
const UCF = UpdatableCholeskyFactorizations
const UQRF = UpdatableQRFactorizations

spd(n) = (B = randn(n, n); Matrix(Symmetric(B * B' + n * I)))

function cholesky_cells(ns; samples::Int = 100)
    for n in ns
        A = spd(n)
        mkF() = UF.UpdatableCholesky(cholesky(Symmetric(A, :L)))
        mkC() = cholesky(Symmetric(A, :L))
        mkU() = UCF.updatable_cholesky(A, 2n)

        # Rank-1 update. The standard library consumes its vector, so it is handed a copy.
        v = randn(n)
        target = A + v * v'
        relerr(F) = norm(Matrix(F) - target) / norm(A)
        F = mkF()
        UF.lowrankupdate!(F, v)
        record!(;
            family = "cholesky", routine = "lowrankupdate!",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @mutating_bench(mkF, F -> UF.lowrankupdate!(F, v), samples = samples),
            relerr = relerr(F),
        )
        C = mkC()
        LinearAlgebra.lowrankupdate!(C, copy(v))
        record!(;
            family = "cholesky", routine = "lowrankupdate!", variant = "LinearAlgebra",
            eltype = Float64, n,
            bench = @mutating_bench(
                mkC, C -> LinearAlgebra.lowrankupdate!(C, copy(v)), samples = samples
            ),
            relerr = relerr(C),
        )
        record!(;
            family = "cholesky", routine = "lowrankupdate!", variant = "recompute",
            eltype = Float64, n,
            bench = @pure_bench(cholesky(Symmetric(target, :L)), samples = samples),
            relerr = relerr(cholesky(Symmetric(target, :L))),
        )

        # Rank-1 downdate. The scaling keeps A - w*w' positive definite at every size.
        w = v ./ (2 * sqrt(n))
        dtarget = A - w * w'
        drelerr(F) = norm(Matrix(F) - dtarget) / norm(A)
        F = mkF()
        UF.lowrankdowndate!(F, w)
        record!(;
            family = "cholesky", routine = "lowrankdowndate!",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @mutating_bench(mkF, F -> UF.lowrankdowndate!(F, w), samples = samples),
            relerr = drelerr(F),
        )
        C = mkC()
        LinearAlgebra.lowrankdowndate!(C, copy(w))
        record!(;
            family = "cholesky", routine = "lowrankdowndate!", variant = "LinearAlgebra",
            eltype = Float64, n,
            bench = @mutating_bench(
                mkC, C -> LinearAlgebra.lowrankdowndate!(C, copy(w)), samples = samples
            ),
            relerr = drelerr(C),
        )
        record!(;
            family = "cholesky", routine = "lowrankdowndate!", variant = "recompute",
            eltype = Float64, n,
            bench = @pure_bench(cholesky(Symmetric(dtarget, :L)), samples = samples),
            relerr = drelerr(cholesky(Symmetric(dtarget, :L))),
        )

        # Symmetric deletion of index 2.
        keep = [i for i in 1:n if i != 2]
        del = A[keep, keep]
        F = mkF()
        UF.delete_column!(F, 2)
        record!(;
            family = "cholesky", routine = "delete_column!",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @mutating_bench(mkF, F -> UF.delete_column!(F, 2), samples = samples),
            relerr = norm(Matrix(F) - del) / norm(A),
        )
        U = mkU()
        UCF.remove_column!(U, 2)
        record!(;
            family = "cholesky", routine = "delete_column!",
            variant = "UpdatableCholeskyFactorizations", eltype = Float64, n,
            bench = @mutating_bench(mkU, U -> UCF.remove_column!(U, 2), samples = samples),
            relerr = norm(Matrix(U) - del) / norm(A),
        )
        record!(;
            family = "cholesky", routine = "delete_column!", variant = "recompute",
            eltype = Float64, n,
            bench = @pure_bench(cholesky(Symmetric(del, :L)), samples = samples),
            relerr = norm(Matrix(cholesky(Symmetric(del, :L))) - del) / norm(A),
        )

        # Appending an index at the end, and inserting one in the middle. `Ag` is the same matrix
        # in both cases; the middle case permutes it so the inserted index lands at position 2.
        Bg = randn(n + 1, n + 1)
        Ag = Matrix(Symmetric(Bg * Bg' + (n + 1) * I))
        x = Ag[:, n + 1]
        mkFg() = UF.UpdatableCholesky(cholesky(Symmetric(Ag[1:n, 1:n], :L)))
        mkUg() = UCF.updatable_cholesky(Ag[1:n, 1:n], 2n + 2)
        F = mkFg()
        UF.insert_column!(F, n + 1, x)
        record!(;
            family = "cholesky", routine = "append_column!",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @mutating_bench(mkFg, F -> UF.insert_column!(F, n + 1, x), samples = samples),
            relerr = norm(Matrix(F) - Ag) / norm(Ag),
        )
        U = mkUg()
        UCF.add_column!(U, x)
        record!(;
            family = "cholesky", routine = "append_column!",
            variant = "UpdatableCholeskyFactorizations", eltype = Float64, n,
            bench = @mutating_bench(mkUg, U -> UCF.add_column!(U, x), samples = samples),
            relerr = norm(Matrix(U) - Ag) / norm(Ag),
        )
        record!(;
            family = "cholesky", routine = "append_column!", variant = "recompute",
            eltype = Float64, n,
            bench = @pure_bench(cholesky(Symmetric(Ag, :L)), samples = samples),
            relerr = norm(Matrix(cholesky(Symmetric(Ag, :L))) - Ag) / norm(Ag),
        )

        into2 = [1; n + 1; 2:n]
        Ai = Ag[into2, into2]
        xi = x[[1; n + 1; 2:n]]
        F = mkFg()
        UF.insert_column!(F, 2, xi)
        record!(;
            family = "cholesky", routine = "insert_column!",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @mutating_bench(mkFg, F -> UF.insert_column!(F, 2, xi), samples = samples),
            relerr = norm(Matrix(F) - Ai) / norm(Ai),
        )
        record!(;
            family = "cholesky", routine = "insert_column!", variant = "recompute",
            eltype = Float64, n,
            bench = @pure_bench(cholesky(Symmetric(Ai, :L)), samples = samples),
            relerr = norm(Matrix(cholesky(Symmetric(Ai, :L))) - Ai) / norm(Ai),
        )

        # Moving index 1 to the end, the widest shift the size admits.
        p = [2:n; 1]
        sh = A[p, p]
        F = mkF()
        UF.shift_columns!(F, 1, n)
        record!(;
            family = "cholesky", routine = "shift_columns!",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @mutating_bench(mkF, F -> UF.shift_columns!(F, 1, n), samples = samples),
            relerr = norm(Matrix(F) - sh) / norm(A),
        )
        record!(;
            family = "cholesky", routine = "shift_columns!", variant = "recompute",
            eltype = Float64, n,
            bench = @pure_bench(cholesky(Symmetric(sh, :L)), samples = samples),
            relerr = norm(Matrix(cholesky(Symmetric(sh, :L))) - sh) / norm(A),
        )
    end
    return
end

function lu_cells(ns; samples::Int = 100)
    for n in ns
        A = Matrix(randn(n, n) + n * I)
        u = randn(n)
        v = randn(n)
        target = A + u * v'

        # Unpivoted. QRupdatesFast.lu1up! overwrites both factors and takes L as m x n and R as
        # n x n, so each sample is handed freshly materialized copies.
        Gn = lu(A, NoPivot())
        mkFn() = UF.UpdatableLU(lu(A, NoPivot()))
        mkQn() = (Matrix(Gn.L), Matrix(Gn.U))
        F = mkFn()
        UF.lowrankupdate!(F, u, v)
        record!(;
            family = "lu", routine = "lowrankupdate! nopivot",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @mutating_bench(mkFn, F -> UF.lowrankupdate!(F, u, v), samples = samples),
            relerr = norm(Matrix(F) - target) / norm(A),
        )
        Ln, Rn = mkQn()
        QRupdatesFast.lu1up!(Ln, Rn, copy(u), copy(v))
        record!(;
            family = "lu", routine = "lowrankupdate! nopivot", variant = "QRupdatesFast",
            eltype = Float64, n,
            bench = @mutating_bench(
                mkQn, t -> QRupdatesFast.lu1up!(t[1], t[2], copy(u), copy(v)), samples = samples
            ),
            relerr = norm(Ln * Rn - target) / norm(A),
        )
        record!(;
            family = "lu", routine = "lowrankupdate! nopivot", variant = "recompute",
            eltype = Float64, n,
            bench = @pure_bench(lu(target, NoPivot()), samples = samples),
            relerr = norm(Matrix(lu(target, NoPivot())) - target) / norm(A),
        )

        # Pivoted. lup1up!'s permutation argument is in/out: the identity that holds afterwards
        # uses the vector it returns, in the same convention as LinearAlgebra.lu.
        Gp = lu(A)
        mkFp() = UF.UpdatableLU(lu(A))
        mkQp() = (Matrix(Gp.L), Matrix(Gp.U), Int32.(Gp.p))
        F = mkFp()
        UF.lowrankupdate!(F, u, v)
        record!(;
            family = "lu", routine = "lowrankupdate! pivoted",
            variant = "UpdatableFactorizations", eltype = Float64, n,
            bench = @mutating_bench(mkFp, F -> UF.lowrankupdate!(F, u, v), samples = samples),
            relerr = norm(Matrix(F) - target) / norm(A),
        )
        Lp, Rp, pp = mkQp()
        QRupdatesFast.lup1up!(Lp, Rp, pp, copy(u), copy(v))
        record!(;
            family = "lu", routine = "lowrankupdate! pivoted", variant = "QRupdatesFast",
            eltype = Float64, n,
            bench = @mutating_bench(
                mkQp, t -> QRupdatesFast.lup1up!(t[1], t[2], t[3], copy(u), copy(v)),
                samples = samples
            ),
            relerr = norm(Lp * Rp - target[Int.(pp), :]) / norm(A),
        )
        record!(;
            family = "lu", routine = "lowrankupdate! pivoted", variant = "recompute",
            eltype = Float64, n,
            bench = @pure_bench(lu(target), samples = samples),
            relerr = norm(Matrix(lu(target)) - target) / norm(A),
        )
    end
    return
end

# Reconstruction residual of a thin QR against the matrix it should factor.
qrerr(Q, R, A) = norm(Q * R - A) / norm(A)
qrerr(F::UF.UpdatableQR, A) = norm(Matrix(F) - A) / norm(A)

# QRupdate maintains R alone, from the normal equations, so its residual is measured on R'R.
rerr(R, A) = norm(R' * R - A' * A) / norm(A' * A)

const RONLY = Dict{String, Any}("r_only" => true)
const FULLQ = Dict{String, Any}("full_q" => true)

function qr_cells(sizes; samples::Int = 100)
    for (m, n) in sizes
        A = randn(m, n)
        Q0 = Matrix(qr(A).Q)
        R0 = Matrix(qr(A).R)
        mkF() = UF.UpdatableQR(A)

        # Rank-1 update.
        u = randn(m)
        v = randn(n)
        target = A + u * v'
        F = mkF()
        UF.lowrankupdate!(F, u, v)
        record!(;
            family = "qr", routine = "lowrankupdate!", variant = "UpdatableFactorizations",
            eltype = Float64, m, n,
            bench = @mutating_bench(mkF, F -> UF.lowrankupdate!(F, u, v), samples = samples),
            relerr = qrerr(F, target),
        )
        record!(;
            family = "qr", routine = "lowrankupdate!", variant = "recompute", eltype = Float64,
            m, n,
            bench = @pure_bench(qr(target), samples = samples),
            relerr = qrerr(Matrix(qr(target).Q), Matrix(qr(target).R), target),
        )

        # Column insertion at the end.
        x = randn(m)
        wide = [A x]
        F = mkF()
        UF.insert_column!(F, n + 1, x)
        record!(;
            family = "qr", routine = "insert_column!", variant = "UpdatableFactorizations",
            eltype = Float64, m, n,
            bench = @mutating_bench(mkF, F -> UF.insert_column!(F, n + 1, x), samples = samples),
            relerr = qrerr(F, wide),
        )
        mkG() = UQRF.UpdatableGivensQR(A, n + 1)
        G = mkG()
        UQRF.add_column!(G, x)
        record!(;
            family = "qr", routine = "insert_column!", variant = "UpdatableQRFactorizations",
            eltype = Float64, m, n,
            bench = @mutating_bench(mkG, G -> UQRF.add_column!(G, x), samples = samples),
            relerr = qrerr(Matrix(G.Q)[:, 1:size(G.R, 1)], G.R, wide), extra = FULLQ,
        )
        record!(;
            family = "qr", routine = "insert_column!", variant = "QRupdate", eltype = Float64,
            m, n,
            bench = @pure_bench(QRupdate.qraddcol(A, R0, x), samples = samples),
            relerr = rerr(QRupdate.qraddcol(A, R0, x), wide), extra = RONLY,
        )
        record!(;
            family = "qr", routine = "insert_column!", variant = "recompute", eltype = Float64,
            m, n,
            bench = @pure_bench(qr(wide), samples = samples),
            relerr = qrerr(Matrix(qr(wide).Q), Matrix(qr(wide).R), wide),
        )

        # Column deletion.
        keepc = [j for j in 1:n if j != 2]
        narrow = A[:, keepc]
        F = mkF()
        UF.delete_column!(F, 2)
        record!(;
            family = "qr", routine = "delete_column!", variant = "UpdatableFactorizations",
            eltype = Float64, m, n,
            bench = @mutating_bench(mkF, F -> UF.delete_column!(F, 2), samples = samples),
            relerr = qrerr(F, narrow),
        )
        mkG2() = UQRF.UpdatableGivensQR(A, n)
        G = mkG2()
        UQRF.remove_column!(G, 2)
        record!(;
            family = "qr", routine = "delete_column!", variant = "UpdatableQRFactorizations",
            eltype = Float64, m, n,
            bench = @mutating_bench(mkG2, G -> UQRF.remove_column!(G, 2), samples = samples),
            relerr = qrerr(Matrix(G.Q)[:, 1:size(G.R, 1)], G.R, narrow), extra = FULLQ,
        )
        record!(;
            family = "qr", routine = "delete_column!", variant = "QRupdate", eltype = Float64,
            m, n,
            bench = @pure_bench(QRupdate.qrdelcol(R0, 2), samples = samples),
            relerr = rerr(QRupdate.qrdelcol(R0, 2), narrow), extra = RONLY,
        )
        record!(;
            family = "qr", routine = "delete_column!", variant = "recompute", eltype = Float64,
            m, n,
            bench = @pure_bench(qr(narrow), samples = samples),
            relerr = qrerr(Matrix(qr(narrow).Q), Matrix(qr(narrow).R), narrow),
        )

        # Column shift: move column 1 to the last position.
        pc = [2:n; 1]
        shifted = A[:, pc]
        F = mkF()
        UF.shift_columns!(F, 1, n)
        record!(;
            family = "qr", routine = "shift_columns!", variant = "UpdatableFactorizations",
            eltype = Float64, m, n,
            bench = @mutating_bench(mkF, F -> UF.shift_columns!(F, 1, n), samples = samples),
            relerr = qrerr(F, shifted),
        )
        mkQR() = (copy(Q0), copy(R0))
        Qs, Rs = mkQR()
        QRupdatesFast.qrshift!(Qs, Rs, 1, n)
        record!(;
            family = "qr", routine = "shift_columns!", variant = "QRupdatesFast",
            eltype = Float64, m, n,
            bench = @mutating_bench(
                mkQR, t -> QRupdatesFast.qrshift!(t[1], t[2], 1, n), samples = samples
            ),
            relerr = qrerr(Qs, Rs, shifted),
        )
        record!(;
            family = "qr", routine = "shift_columns!", variant = "recompute", eltype = Float64,
            m, n,
            bench = @pure_bench(qr(shifted), samples = samples),
            relerr = qrerr(Matrix(qr(shifted).Q), Matrix(qr(shifted).R), shifted),
        )

        # Row insertion at the end.
        y = randn(n)
        tall = [A; transpose(y)]
        F = mkF()
        UF.insert_row!(F, m + 1, y)
        record!(;
            family = "qr", routine = "insert_row!", variant = "UpdatableFactorizations",
            eltype = Float64, m, n,
            bench = @mutating_bench(mkF, F -> UF.insert_row!(F, m + 1, y), samples = samples),
            relerr = qrerr(F, tall),
        )
        record!(;
            family = "qr", routine = "insert_row!", variant = "QRupdate", eltype = Float64, m, n,
            bench = @pure_bench(QRupdate.qraddrow(R0, reshape(y, 1, n)), samples = samples),
            relerr = rerr(QRupdate.qraddrow(R0, reshape(y, 1, n)), tall), extra = RONLY,
        )
        record!(;
            family = "qr", routine = "insert_row!", variant = "recompute", eltype = Float64, m, n,
            bench = @pure_bench(qr(tall), samples = samples),
            relerr = qrerr(Matrix(qr(tall).Q), Matrix(qr(tall).R), tall),
        )

        # Row deletion. A random Gaussian A has no leverage-one row, so the default rtol never
        # fires here; the failure path is the test suite's, not the benchmark's.
        keepr = [i for i in 1:m if i != 2]
        short = A[keepr, :]
        F = mkF()
        UF.delete_row!(F, 2)
        record!(;
            family = "qr", routine = "delete_row!", variant = "UpdatableFactorizations",
            eltype = Float64, m, n,
            bench = @mutating_bench(mkF, F -> UF.delete_row!(F, 2), samples = samples),
            relerr = qrerr(F, short),
        )
        record!(;
            family = "qr", routine = "delete_row!", variant = "recompute", eltype = Float64, m, n,
            bench = @pure_bench(qr(short), samples = samples),
            relerr = qrerr(Matrix(qr(short).Q), Matrix(qr(short).R), short),
        )
    end
    return
end
