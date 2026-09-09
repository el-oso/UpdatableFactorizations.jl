module UpdatableFactorizations

using LinearAlgebra
using LinearAlgebra: givensAlgorithm, Givens, PosDefException, ZeroPivotException, QRCompactWY
import LinearAlgebra: lowrankupdate!, lowrankdowndate!, ldiv!, logdet, det
using TypeContracts
# `@strict` (type-stability + no owned-scratch dictionary lookups + allocation-freedom) guards
# the QR, Cholesky and LU rank-1 kernels below. With `StrictMode.checks_enabled()` true (the
# default, and the state ordinary development and CI run in), each guarded call's own
# reflection has a real, fixed cost — several kilobytes and a dynamic dispatch — so
# `@allocated`/`@test_noalloc`/`@test_typestable` measure that reflection, not the kernel, and
# a raw allocation or JET check against the wrapped call is expected to see it. The guarantee
# only reduces to the bare kernel call, with none of that cost, once `checks_enabled` is false —
# the configuration a shipped build sets.
using StrictMode

export UpdatableCholesky, UpdatableLU, UpdatableQR
export insert_column!, delete_column!, shift_columns!, insert_row!, delete_row!
export qr_householder
export cholesky_crout, lu_crout, qr_bcgs
public AbstractQRep, DenseQ, materialize, capacity
public default_rankk!

include("cholesky_type.jl")
include("cholesky_update.jl")
include("cholesky_resize.jl")
include("lu_type.jl")
include("lu_update.jl")
include("qr_rep.jl")
include("qr_type.jl")
include("qr_update.jl")
include("construct.jl")
include("contracts.jl")

end
