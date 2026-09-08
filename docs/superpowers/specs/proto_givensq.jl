using LinearAlgebra
using LinearAlgebra: givensAlgorithm, Givens

mutable struct GQ{T, R <: Real, B}
    base::B
    plane::Vector{Int}
    cosines::Vector{R}
    sines::Vector{T}
    work::Vector{T}
    m::Int
    n::Int
end

function GQ(F::LinearAlgebra.QRCompactWY{T}) where {T}
    m, n = size(F)
    R = real(T)
    return GQ{T, R, typeof(F.Q)}(F.Q, Int[], R[], T[], zeros(T, m), m, n)
end

nrot(Q::GQ) = length(Q.plane)

# rotation k of the log, as a Givens acting in the (plane[k], plane[k]+1) plane
_rot(Q::GQ{T}, k::Int) where {T} =
    Givens(Q.plane[k], Q.plane[k] + 1, T(Q.cosines[k]), Q.sines[k])
_rotadj(Q::GQ{T}, k::Int) where {T} =
    Givens(Q.plane[k], Q.plane[k] + 1, T(Q.cosines[k]), -Q.sines[k])

Base.size(Q::GQ) = (Q.m, Q.n)

function push_rot!(Q::GQ, G::Givens)
    G.i2 == G.i1 + 1 || error("adjacent planes only")
    push!(Q.plane, G.i1)
    push!(Q.cosines, real(G.c))
    push!(Q.sines, G.s)
    return Q
end

# y = Q * x, x length n, y length m
function apply!(y, Q::GQ, x)
    copyto!(view(y, 1:Q.n), x)
    fill!(view(y, (Q.n + 1):Q.m), zero(eltype(y)))
    for k in nrot(Q):-1:1
        lmul!(_rot(Q, k), y)
    end
    lmul!(Q.base, y)
    return y
end

# y = Q' * z, z length m, y length n
function applyadj!(y, Q::GQ, z)
    t = Q.work
    copyto!(t, z)
    lmul!(Q.base', t)
    for k in 1:nrot(Q)
        lmul!(_rotadj(Q, k), t)
    end
    copyto!(y, view(t, 1:Q.n))
    return y
end

function dense(Q::GQ{T}) where {T}
    M = zeros(T, Q.m, Q.n)
    for i in 1:Q.n
        M[i, i] = one(T)
    end
    for k in nrot(Q):-1:1
        lmul!(_rot(Q, k), M)
    end
    lmul!(Q.base, M)
    return M
end

function fullmat(Q::GQ{T}) where {T}
    M = Matrix{T}(I, Q.m, Q.m)
    for k in nrot(Q):-1:1
        lmul!(_rot(Q, k), M)
    end
    lmul!(Q.base, M)
    return M
end

# append a column q with q*beta == r
function augment!(Q::GQ{T}, r) where {T}
    Q.n < Q.m || error("no room")
    c = Q.work
    copyto!(c, r)
    lmul!(Q.base', c)
    for k in 1:nrot(Q)
        lmul!(_rotadj(Q, k), c)
    end
    for i in (Q.m - 1):-1:(Q.n + 1)
        H, _ = LinearAlgebra.givens(c[i], c[i + 1], i, i + 1)
        lmul!(H, c)
        push_rot!(Q, Givens(H.i1, H.i2, T(real(H.c)), -H.s))
    end
    beta = c[Q.n + 1]
    Q.n += 1
    return beta
end

function check(T, m, n)
    A = randn(T, m, n)
    F = qr(A)
    Q = GQ(F)
    Qe = Matrix(F.Q)                       # explicit thin, m x n
    x = randn(T, n); y = zeros(T, m)
    @assert norm(apply!(y, Q, x) - Qe * x) < 1e-12
    z = randn(T, m); w = zeros(T, n)
    @assert norm(applyadj!(w, Q, z) - Qe' * z) < 1e-12
    @assert norm(dense(Q) - Qe) < 1e-12

    # rmul! by a Givens inside 1:n matches rotating the explicit thin factor
    Qe2 = copy(Qe)
    if n >= 2
        G, _ = LinearAlgebra.givens(randn(T), randn(T), n - 1, n)
        push_rot!(Q, G)
        Qe2 = rmul!(copy(Qe), G)
        @assert norm(dense(Q) - Qe2) < 1e-11 (T, m, n, norm(dense(Q) - Qe2))
    end

    # augment: q * beta == r for r orthogonal to range(Q)
    u = randn(T, m)
    w2 = zeros(T, n); applyadj!(w2, Q, u)
    y2 = zeros(T, m); apply!(y2, Q, w2)
    r = u - y2
    beta = augment!(Q, r)
    D = dense(Q)                            # m x (n+1)
    @assert norm(D[:, n + 1] * beta - r) < 1e-11 (T, m, n, norm(D[:, n+1]*beta - r))
    @assert norm(D' * D - I) < 1e-11
    @assert abs(abs(beta) - norm(r)) < 1e-11
    @assert norm(D[:, 1:n] - Qe2) < 1e-11
    # full matrix stays orthogonal
    Fm = fullmat(Q)
    @assert norm(Fm' * Fm - I) < 1e-11
    return true
end

for T in (Float64, ComplexF64), (m, n) in ((8, 3), (12, 5), (6, 5), (20, 1))
    check(T, m, n)
end
println("all good")

# Shrink then re-augment: the dropped column stays in the implicit complement, so a later
# augmentation rotates that complement onto the new residual direction.
function check2(T, m, n)
    A = randn(T, m, n); Q = GQ(qr(A)); Qe = Matrix(qr(A).Q)
    Q.n = n - 1
    @assert norm(dense(Q) - Qe[:, 1:(n - 1)]) < 1e-11
    u = randn(T, m)
    w = zeros(T, n - 1); applyadj!(w, Q, u)
    y = zeros(T, m); apply!(y, Q, w)
    r = u - y
    b = augment!(Q, r)
    D = dense(Q)
    @assert norm(D[:, n] * b - r) < 1e-10
    @assert norm(D' * D - I) < 1e-10
    @assert norm(D[:, 1:(n - 1)] - Qe[:, 1:(n - 1)]) < 1e-10
    @assert abs(abs(b) - norm(r)) < 1e-10
    return true
end

for T in (Float64, ComplexF64), (m, n) in ((10, 4), (7, 7), (9, 2))
    check2(T, m, n)
end
println("shrink/augment good")
