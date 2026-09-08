module UpdatableFactorizations

using LinearAlgebra
using LinearAlgebra: givensAlgorithm, PosDefException, ZeroPivotException
import LinearAlgebra: lowrankupdate!, lowrankdowndate!, ldiv!, logdet, det

export UpdatableCholesky, UpdatableLU
export insert_column!, delete_column!, shift_columns!

include("cholesky_type.jl")
include("cholesky_update.jl")
include("cholesky_resize.jl")
include("lu_type.jl")
include("lu_update.jl")
include("contracts.jl")

end
