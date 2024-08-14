module RecurrenceRelationshipsFillArraysExt
using RecurrenceRelationships, FillArrays
using FillArrays: AbstractFill, getindex_value

import RecurrenceRelationships: _forwardrecurrence_next, _clenshaw_next

# special case for B[n] == 0
Base.@propagate_inbounds _forwardrecurrence_next(n, A, ::Zeros, C, x, p0, p1) = muladd(A[n]*x, p1, -C[n]*p0)
# special case for Chebyshev U
Base.@propagate_inbounds _forwardrecurrence_next(n, A::AbstractFill, ::Zeros, C::Ones, x, p0, p1) = muladd(getindex_value(A)*x, p1, -p0)

Base.@propagate_inbounds _clenshaw_next(n, A, ::Zeros, C, x, c, bn1, bn2) = muladd(A[n]*x, bn1, muladd(-C[n+1],bn2,c[n]))
# Chebyshev U
Base.@propagate_inbounds _clenshaw_next(n, A::AbstractFill, ::Zeros, C::Ones, x, c, bn1, bn2) = muladd(getindex_value(A)*x, bn1, -bn2+c[n])

end