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
