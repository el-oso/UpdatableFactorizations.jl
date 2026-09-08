using TypeContracts

@invariants UpdatableCholesky begin
    "size is within capacity" => F -> 0 <= F.n <= size(F.factors, 1)
    "uplo is L or U" => F -> F.uplo == 'L' || F.uplo == 'U'
    "workspaces cover the active block" =>
        F -> length(F.work) >= F.n && length(F.cosines) >= F.n && length(F.rot) >= F.n
    "the stored factor has a positive diagonal" =>
        F -> all(i -> real(F.factors[i, i]) > 0, 1:F.n)
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
end
