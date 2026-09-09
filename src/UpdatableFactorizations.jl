module UpdatableFactorizations

using LinearAlgebra
using LinearAlgebra: givensAlgorithm, Givens, PosDefException, ZeroPivotException, QRCompactWY
import LinearAlgebra: lowrankupdate!, lowrankdowndate!, ldiv!, logdet, det
using TypeContracts
# `@assert_noalloc`/`@assert_typestable` are not wrapped around the QR, Cholesky or LU rank-1
# kernels: each guarantee's own call-site bookkeeping (`Base.return_types`, the allocation-signal
# scan) allocates several kilobytes and dispatches dynamically, which is exactly what AllocCheck
# and JET catch when the wrapped kernel is itself checked for allocation-freedom or type
# stability. Wrapping one of those kernels turns its own `@test_noalloc`/`@test_typestable` gate
# from passing to failing.
using StrictMode

export UpdatableCholesky, UpdatableLU, UpdatableQR
export insert_column!, delete_column!, shift_columns!, insert_row!, delete_row!
export qr_householder
public AbstractQRep, DenseQ, materialize, capacity

include("cholesky_type.jl")
include("cholesky_update.jl")
include("cholesky_resize.jl")
include("lu_type.jl")
include("lu_update.jl")
include("qr_rep.jl")
include("qr_type.jl")
include("qr_update.jl")
include("contracts.jl")

end
