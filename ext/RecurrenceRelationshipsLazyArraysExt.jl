module RecurrenceRelationshipsLazyArraysExt
using RecurrenceRelationships, LazyArrays
using LazyArrays.FillArrays

import RecurrenceRelationships: forwardrecurrence_next, clenshaw_next
import LazyArrays.FillArrays: AbstractFill

##
# For Chebyshev T. Note the shift in indexing is fine due to the AbstractFill
##
Base.@propagate_inbounds forwardrecurrence_next(n, A::Vcat{<:Any,1,<:Tuple{<:Number,<:AbstractFill}}, B::Zeros, C::Ones, x, p0, p1) = 
    forwardrecurrence_next(n, A.args[2], B, C, x, p0, p1)


Base.@propagate_inbounds clenshaw_next(n, A::Vcat{<:Any,1,<:Tuple{<:Number,<:AbstractFill}}, B::Zeros, C::Ones, x, c, bn1, bn2) = 
    clenshaw_next(n, A.args[2], B, C, x, c, bn1, bn2)
end