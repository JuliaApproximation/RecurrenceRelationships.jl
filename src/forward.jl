"""
   forwardrecurrence!(v, A, B, C, x)

evaluates the orthogonal polynomials at points `x`,
where `A`, `B`, and `C` are `AbstractVector`s containing the recurrence coefficients
as defined in DLMF,
overwriting `v` with the results.
"""
function forwardrecurrence!(v::AbstractVector{T}, A::AbstractVector, B::AbstractVector, C::AbstractVector, x, p0=one(T)) where T
    N = length(v)
    N == 0 && return v
    length(A)+1 ≥ N && length(B)+1 ≥ N && length(C)+1 ≥ N || throw(ArgumentError("A, B, C must contain at least $(N-1) entries"))
    p1 = convert(T, N == 1 ? p0 : muladd(A[1],x,B[1])*p0) # avoid accessing A[1]/B[1] if empty
    _forwardrecurrence!(v, A, B, C, x, convert(T, p0), p1)
end


Base.@propagate_inbounds _forwardrecurrence_next(n, A, B, C, x, p0, p1) = muladd(muladd(A[n],x,B[n]), p1, -C[n]*p0)
# special case for B[n] == 0
Base.@propagate_inbounds _forwardrecurrence_next(n, A, ::Zeros, C, x, p0, p1) = muladd(A[n]*x, p1, -C[n]*p0)
# special case for Chebyshev U
Base.@propagate_inbounds _forwardrecurrence_next(n, A::AbstractFill, ::Zeros, C::Ones, x, p0, p1) = muladd(getindex_value(A)*x, p1, -p0)


# this supports adaptivity: we can populate `v` for large `n`
function _forwardrecurrence!(v::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x, p0, p1)
    N = length(v)
    N == 0 && return v
    v[1] = p0
    N == 1 && return v
    v[2] = p1
    _forwardrecurrence!(v, A, B, C, x, 2:N)
end

function _forwardrecurrence!(v::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x, kr::AbstractUnitRange)
    n₀, N = first(kr), last(kr)
    @boundscheck N > length(v) && throw(BoundsError(v, N))
    p0, p1 = v[n₀-1], v[n₀]
    @inbounds for n = n₀:N-1
        p1,p0 = _forwardrecurrence_next(n, A, B, C, x, p0, p1),p1
        v[n+1] = p1
    end
    v
end




forwardrecurrence(N::Integer, A::AbstractVector, B::AbstractVector, C::AbstractVector, x) =
    forwardrecurrence!(Vector{promote_type(eltype(A),eltype(B),eltype(C),typeof(x))}(undef, N), A, B, C, x)



##
# For Chebyshev T. Note the shift in indexing is fine due to the AbstractFill
##
Base.@propagate_inbounds _forwardrecurrence_next(n, A::Vcat{<:Any,1,<:Tuple{<:Number,<:AbstractFill}}, B::Zeros, C::Ones, x, p0, p1) = 
    _forwardrecurrence_next(n, A.args[2], B, C, x, p0, p1)

function initiateforwardrecurrence(N, A, B, C, x, μ)
    T = promote_type(eltype(A), eltype(B), eltype(C), typeof(x))
    p0 = convert(T, μ)
    N == 0 && return zero(T), p0
    p1 = convert(T, muladd(A[1],x,B[1])*p0)
    @inbounds for n = 2:N
        p1,p0 = _forwardrecurrence_next(n, A, B, C, x, p0, p1),p1
    end
    p0,p1
end
