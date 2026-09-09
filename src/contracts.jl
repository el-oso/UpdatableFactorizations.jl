@invariants UpdatableCholesky begin
    "size is within capacity" => F -> 0 <= F.n <= size(F.factors, 1)
    "workspaces cover the active block" =>
        F -> length(F.work) >= F.n && length(F.cosines) >= F.n &&
        length(F.rot) >= F.n && length(F.perm) >= F.n
    "the active block is lower triangular" =>
        F -> all(iszero, [F.factors[i, j] for j in 1:F.n for i in 1:(j - 1)])
    "storage outside the active block is zero" =>
        F -> all(iszero, view(F.factors, (F.n + 1):size(F.factors, 1), :)) &&
        all(iszero, view(F.factors, :, (F.n + 1):size(F.factors, 2)))
    "the stored factor has a real positive diagonal" =>
        F -> all(i -> isreal(F.factors[i, i]) && real(F.factors[i, i]) > 0, 1:F.n)
end

# UpdatableLU overrides getproperty for :L and :U, so these predicates read the underlying
# fields with getfield to avoid reassembling those matrices on every check.
@invariants UpdatableLU begin
    "factors agree in size" =>
        F -> length(getfield(F, :d)) == size(getfield(F, :Lf), 1) == size(getfield(F, :Uf), 1)
    # p is a dense 1-based LAPACK pivot vector, so comparing against the literal range 1:n
    # is the permutation check itself, not a stand-in for iterating p's own indices.
    "p is a permutation" => F -> sort(getfield(F, :p)) == 1:length(getfield(F, :d)) # noidiom
    "workspace holds both update vectors" =>
        F -> length(getfield(F, :work)) >= 2length(getfield(F, :d))
    "Lf is unit lower triangular" =>
        F -> getfield(F, :Lf) == UnitLowerTriangular(getfield(F, :Lf))
    "Uf is unit upper triangular" =>
        F -> getfield(F, :Uf) == UnitUpperTriangular(getfield(F, :Uf))
end

@invariants UpdatableQR begin
    # The body is parenthesized: an unparenthesized `;` splits the clause into two block
    # statements and the macro rejects the second as a spec that is not a pair.
    "size is within capacity" =>
        F -> ((mc, nc) = capacity(F); 0 <= F.n <= F.m <= mc && F.n <= nc)
    "the representation agrees with the factorization" =>
        F -> size(getfield(F, :qrep)) == (F.m, F.n)
    "workspaces cover the augmented block" =>
        F -> length(F.work) >= F.n + 1 && length(F.corr) >= F.n + 1
    "the active block of R is upper triangular" =>
        F -> all(iszero, [F.factors[i, j] for j in 1:F.n for i in (j + 1):F.n])
    "storage outside the active blocks is zero" =>
        F -> (
        Q = getfield(F, :qrep).buf; R = F.factors;
        all(iszero, view(Q, (F.m + 1):size(Q, 1), :)) &&
            all(iszero, view(Q, :, (F.n + 1):size(Q, 2))) &&
            all(iszero, view(R, (F.n + 1):size(R, 1), :)) &&
            all(iszero, view(R, :, (F.n + 1):size(R, 2)))
    )
    # Comparing the Gram matrix against an explicit identity over the active columns is the
    # assertion itself, not a stand-in for iterating the factor's own indices.
    "Q has orthonormal columns" =>                                              # noidiom
        F -> (Q = F.Q; norm(Q' * Q - I) <= sqrt(eps(real(eltype(Q)))) * max(F.n, 1))
end

@strict_contract AbstractUpdatableCholesky "the verb surface an updatable Cholesky factorization exposes" begin
    lowrankupdate!(::Self, v::AbstractVector)::Self => "replace A with A + v*v'"
    lowrankdowndate!(::Self, v::AbstractVector)::Self => "replace A with A - v*v'"
    insert_column!(::Self, j::Integer, x::AbstractVector)::Self => "insert row and column j"
    delete_column!(::Self, j::Integer)::Self => "delete row and column j"
    shift_columns!(::Self, i::Integer, j::Integer)::Self => "move index i to position j"
    size(::Self)::Tuple{Int, Int} => "current size"
end

@strict_contract AbstractUpdatableLU "the verb surface an updatable LU factorization exposes" begin
    lowrankupdate!(::Self, u::AbstractVector, v::AbstractVector)::Self =>
        "replace A with A + u*v'"
    size(::Self)::Tuple{Int, Int} => "current size"
end

@strict_contract AbstractUpdatableQR "the verb surface an updatable QR factorization exposes" begin
    lowrankupdate!(::Self, u::AbstractVector, v::AbstractVector)::Self =>
        "replace A with A + u*v'"
    insert_column!(::Self, j::Integer, x::AbstractVector)::Self => "insert column j"
    delete_column!(::Self, j::Integer)::Self => "delete column j"
    shift_columns!(::Self, i::Integer, j::Integer)::Self => "move column i to position j"
    insert_row!(::Self, i::Integer, x::AbstractVector)::Self => "insert row i"
    delete_row!(::Self, i::Integer)::Self => "delete row i"
    size(::Self)::Tuple{Int, Int} => "current shape"
    capacity(::Self)::Tuple{Int, Int} => "largest shape before reallocation"
end
