module RecurrenceRelationshipsStaticArraysCoreExt
using RecurrenceRelationships, StaticArraysCore
using StaticArraysCore: StaticVector
import RecurrenceRelationships: clenshaw, check_clenshaw_recurrences

# Views of matrices where the columns (or rows) are chosen by a static vector, e.g. view(c, :, SVector(2,3))
const StaticColumnsOrRowsView = SubArray{<:Any,2,<:AbstractMatrix,<:Union{Tuple{Any,StaticVector},Tuple{StaticVector,Any}}}

# Evaluate the N expansions whose k-th coefficients are coeff(k, j) for j = 1:N, in a single backward pass that looks
# up each recurrence coefficient once. The state is kept in tuples so nothing is allocated.
function _static_clenshaw(coeff, m, ::Val{N}, ::Type{T}, A, B, C, x) where {N,T}
    @boundscheck check_clenshaw_recurrences(m, A, B, C)
    m == 0 && return SVector{N,T}(ntuple(_ -> zero(T), Val(N)))
    bn1 = ntuple(j -> convert(T, coeff(m, j)), Val(N))
    bn2 = ntuple(_ -> zero(T), Val(N))
    @inbounds for k = m-1:-1:1
        bn1, bn2 = _static_clenshaw_next(coeff, k, Val(N), A, B, C, x, bn1, bn2), bn1
    end
    SVector{N,T}(bn1)
end

# passing bn1 and bn2 as arguments, rather than capturing them in a closure, avoids boxing them
Base.@propagate_inbounds function _static_clenshaw_next(coeff, k, ::Val{N}, A, B, C, x, bn1, bn2) where N
    αₖ, γₖ = muladd(A[k],x,B[k]), C[k+1]
    map((b1, b2, cₖ) -> muladd(αₖ, b1, muladd(-γₖ, b2, cₖ)), bn1, bn2, ntuple(j -> coeff(k, j), Val(N)))
end

"""
    clenshaw(view(c, :, SVector(j...)), A, B, C, x; dims=1)
    clenshaw(view(c, SVector(k...), :), A, B, C, x; dims=2)

evaluate the expansions whose coefficients are the chosen columns (or rows) of `c`, returning an `SVector`.
The recurrence coefficients are only looked up once for all of them and nothing is allocated. Other values of
`dims` fall back to the `AbstractMatrix` method, so the return type is only inferred when `dims` is a constant.
"""
Base.@constprop :aggressive function clenshaw(c::StaticColumnsOrRowsView, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::Number; dims)
    T = promote_type(eltype(c),eltype(A),eltype(B),eltype(C),typeof(x))
    kr, jr = parentindices(c)
    if dims == 1 && jr isa StaticVector
        _static_clenshaw((k, j) -> @inbounds(c[k,j]), size(c,1), Val(length(jr)), T, A, B, C, x)
    elseif dims == 2 && kr isa StaticVector
        _static_clenshaw((k, j) -> @inbounds(c[j,k]), size(c,2), Val(length(kr)), T, A, B, C, x)
    else
        invoke(clenshaw, Tuple{AbstractMatrix,AbstractVector,AbstractVector,AbstractVector,Number}, c, A, B, C, x; dims)
    end
end

end
