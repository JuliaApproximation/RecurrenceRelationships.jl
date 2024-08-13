module RecurrenceRelationshipsFillArraysExt
using RecurrenceRelationships, FillArrays
using RecurrenceRelationships.LinearAlgebra

import RecurrenceRelationships: _forwardrecurrence_next, _clenshaw_next, _clenshaw_next!

# special case for B[n] == 0
Base.@propagate_inbounds _forwardrecurrence_next(n, A, ::Zeros, C, x, p0, p1) = muladd(A[n]*x, p1, -C[n]*p0)
# special case for Chebyshev U
Base.@propagate_inbounds _forwardrecurrence_next(n, A::AbstractFill, ::Zeros, C::Ones, x, p0, p1) = muladd(getindex_value(A)*x, p1, -p0)

Base.@propagate_inbounds _clenshaw_next(n, A, ::Zeros, C, x, c, bn1, bn2) = muladd(A[n]*x, bn1, muladd(-C[n+1],bn2,c[n]))
# Chebyshev U
Base.@propagate_inbounds _clenshaw_next(n, A::AbstractFill, ::Zeros, C::Ones, x, c, bn1, bn2) = muladd(getindex_value(A)*x, bn1, -bn2+c[n])


###
# Operator clenshaw
###


Base.@propagate_inbounds function _clenshaw_next!(n, A::AbstractFill, ::Zeros, C::Ones, x::AbstractMatrix, c, bn1::AbstractMatrix{T}, bn2::AbstractMatrix{T}) where T
    muladd!(getindex_value(A), x, bn1, -one(T), bn2)
    view(bn2,band(0)) .+= c[n]
    bn2
end

Base.@propagate_inbounds function _clenshaw_next!(n, A::AbstractVector, ::Zeros, C::AbstractVector, x::AbstractMatrix, c, bn1::AbstractMatrix{T}, bn2::AbstractMatrix{T}) where T
    muladd!(A[n], x, bn1, -C[n+1], bn2)
    view(bn2,band(0)) .+= c[n]
    bn2
end


# allow special casing first arg, for ChebyshevT in ClassicalOrthogonalPolynomials
Base.@propagate_inbounds function _clenshaw_first!(A, ::Zeros, C, X, c, bn1, bn2) 
    muladd!(A[1], X, bn1, -C[2], bn2)
    view(bn2,band(0)) .+= c[1]
    bn2
end


Base.@propagate_inbounds function _clenshaw_first!(A, ::Zeros, C, X, c, f::AbstractVector, bn1, bn2) 
    muladd!(A[1], X, bn1, -C[2], bn2)
    bn2 .+= c[1] .* f
    bn2
end
end