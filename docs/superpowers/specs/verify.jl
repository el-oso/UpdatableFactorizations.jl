# Exhaustive verification of the prototype kernels. Every operation is checked against a
# factorization recomputed from scratch, for every index, both uplo, real and complex.
include("proto.jl")

const FAILS = String[]

function ck(name, got, want)
    e = norm(got - want) / norm(want)
    (isfinite(e) && e < 1.0e-10) || push!(FAILS, "$name  relerr=$e")
    return e
end

function verify(T, n)
    A = spd(n, T)
    v = randn(T, n)
    for up in (:L, :U)
        F = UChol(cholesky(Hermitian(A, up)))
        lowrankupdate_chol!(F, copy(v))
        ck("ch1up $T uplo=$up", reconstruct(F), A + v * v')

        F = UChol(cholesky(Hermitian(A, up)))
        lowrankdowndate_chol!(F, copy(v) ./ 8)
        ck("ch1dn $T uplo=$up", reconstruct(F), A - (v ./ 8) * (v ./ 8)')

        for j in 1:n
            F = UChol(cholesky(Hermitian(A, up)))
            delete_index!(F, j)
            keep = [k for k in 1:n if k != j]
            ck("chdex $T uplo=$up j=$j", reconstruct(F), A[keep, keep])
        end

        B = spd(n + 1, T)
        for j in 1:(n + 1)
            keep = [k for k in 1:(n + 1) if k != j]
            F = UChol(cholesky(Hermitian(B[keep, keep], up)))
            insert_index!(F, j, B[:, j])
            ck("chinx $T uplo=$up j=$j", reconstruct(F), B)
        end

        for i in 1:n, j in 1:n
            F = UChol(cholesky(Hermitian(A, up)))
            shift_indices!(F, i, j)
            p = cyclicperm(n, i, j)
            ck("chshx $T uplo=$up $i->$j", reconstruct(F), A[p, p])
        end
    end
    return
end

verify(Float64, 7)
verify(ComplexF64, 6)

# Bennett LU rank-1, on the LDU form
function verify_lu(T, n)
    A = randn(T, n, n) + n * I
    u = randn(T, n)
    z = randn(T, n)
    F = lu(A, NoPivot())
    L = Matrix(F.L)
    U = Matrix(F.U)
    d = [U[k, k] for k in 1:n]
    Uu = copy(U)
    for k in 1:n, j in k:n
        Uu[k, j] /= d[k]
    end
    # bennett! works in the transpose form, so A + u*z' is passed as second vector conj(z).
    bennett!(L, d, Uu, copy(u), conj.(z), one(T))
    ck("lu1up $T", L * Diagonal(d) * Uu, A + u * z')

    # Pivoted: P*A = L*U, so P*(A + u*z') = L*U + (P*u)*z'. Bennett runs in the permuted frame
    # and retains the permutation.
    G = lu(A)
    Lp = Matrix(G.L)
    Up = Matrix(G.U)
    dp = [Up[k, k] for k in 1:n]
    Uup = copy(Up)
    for k in 1:n, j in k:n
        Uup[k, j] /= dp[k]
    end
    bennett!(Lp, dp, Uup, u[G.p], conj.(z), one(T))
    ck("lup1up $T", Lp * Diagonal(dp) * Uup, (A + u * z')[G.p, :])
    return
end

verify_lu(Float64, 7)
verify_lu(ComplexF64, 6)

if isempty(FAILS)
    println("ALL PASS")
else
    println(length(FAILS), " FAILURES:")
    foreach(f -> println("  ", f), first(FAILS, 20))
end
