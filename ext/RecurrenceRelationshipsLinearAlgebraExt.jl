module RecurrenceRelationshipsLinearAlgebraExt
using RecurrenceRelationships, LinearAlgebra

import RecurrenceRelationships: olver, forwardrecurrence!

olver(T::Tridiagonal, f, n...; kwds...) = olver(T.dl, T.d, T.du, f, n...; kwds...)
olver(T::SymTridiagonal, f, n...; kwds...) = olver(T.ev, T.dv, T.ev, f, n...; kwds...)

function forwardrecurrence!(v::AbstractVector{T}, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::AbstractMatrix, p0=one(x)) where T
    N = length(v)
    N == 0 && return v
    length(A)+1 ≥ N && length(B)+1 ≥ N && length(C)+1 ≥ N || throw(ArgumentError("A, B, C must contain at least $(N-1) entries"))
    p1 = convert(T, N == 1 ? p0 : muladd(A[1],x,B[1]*I)*p0) # avoid accessing A[1]/B[1] if empty
    forwardrecurrence!(v, A, B, C, x, convert(T, p0), p1)
end


end