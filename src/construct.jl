"""
    default_rankk!(C, A, alpha, beta; uplo = 'L') -> C

Overwrite `C` with `alpha*A*A' + beta*C`. On BLAS element types only the triangle named by
`uplo` is read and written, through `syrk!` for real `A` and `herk!` for complex `A`. For any
other element type both triangles are written, because there is no symmetric rank-k kernel to
call and a full product costs less than assembling one triangle entry by entry.

This is the default deferred flush of `cholesky_crout`.
"""
function default_rankk!(
        C::StridedMatrix{T}, A::StridedMatrix{T}, alpha, beta; uplo::Char = 'L'
    ) where {T <: LinearAlgebra.BlasReal}
    return BLAS.syrk!(uplo, 'N', T(alpha), A, T(beta), C)
end

function default_rankk!(
        C::StridedMatrix{T}, A::StridedMatrix{T}, alpha, beta; uplo::Char = 'L'
    ) where {T <: LinearAlgebra.BlasComplex}
    R = real(T)
    return BLAS.herk!(uplo, 'N', R(real(alpha)), A, R(real(beta)), C)
end

default_rankk!(C, A, alpha, beta; uplo::Char = 'L') = mul!(C, A, A', alpha, beta)
