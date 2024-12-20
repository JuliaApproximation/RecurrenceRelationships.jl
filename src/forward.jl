"""
   forwardrecurrence!(v, A, B, C, x)

evaluates the orthogonal polynomials at point `x`,
where `A`, `B`, and `C` are `AbstractVector`s containing the recurrence coefficients
as defined in [DLMF](https://dlmf.nist.gov/18.9),
overwriting `v` with the results.
"""
function forwardrecurrence!(v::AbstractVector{T}, A::AbstractVector, B::AbstractVector, C::AbstractVector, x=zero(T), p0=one(T)) where T
    N = length(v)
    N == 0 && return v
    length(A)+1 ≥ N && length(B)+1 ≥ N && length(C)+1 ≥ N || throw(ArgumentError("A, B, C must contain at least $(N-1) entries"))
    p1 = convert(T, N == 1 ? p0 : muladd(A[1],x,B[1])*p0) # avoid accessing A[1]/B[1] if empty
    forwardrecurrence!(v, A, B, C, x, convert(T, p0), p1)
end


Base.@propagate_inbounds forwardrecurrence_next(n, A, B, C, x, p0, p1) = muladd(muladd(A[n],x,B[n]), p1, -C[n]*p0)


# this supports adaptivity: we can populate `v` for large `n`
function forwardrecurrence!(v::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x, p0, p1)
    N = length(v)
    N == 0 && return v
    v[1] = p0
    N == 1 && return v
    v[2] = p1
    forwardrecurrence_partial!(v, A, B, C, x, 2:N)
end

function forwardrecurrence_partial!(v::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x, kr::AbstractUnitRange)
    n₀, N = first(kr), last(kr)
    @boundscheck N > length(v) && throw(BoundsError(v, N))
    p0, p1 = v[n₀-1], v[n₀]
    @inbounds for n = n₀:N-1
        p1,p0 = forwardrecurrence_next(n, A, B, C, x, p0, p1),p1
        v[n+1] = p1
    end
    v
end



"""
   forwardrecurrence(N, A, B, C, x)

evaluates the first `N` orthogonal polynomials at point `x`,
where `A`, `B`, and `C` are `AbstractVector`s containing the first form recurrence coefficients as defined in 
[DLMF](https://dlmf.nist.gov/18.9), i.e. it returns Pᵢ(x) for i = 0, 1, ..., N-1
"""
forwardrecurrence(N::Integer, A::AbstractVector, B::AbstractVector, C::AbstractVector, x) =
    forwardrecurrence!(Vector{polynomialtype(eltype(A),eltype(B),eltype(C),typeof(x))}(undef, N), A, B, C, x)

forwardrecurrence(N::Integer, A::AbstractVector, B::AbstractVector, C::AbstractVector) =
    forwardrecurrence!(Vector{polynomialtype(eltype(A),eltype(B),eltype(C))}(undef, N), A, B, C)


"""
   forwardrecurrence(A, B, C, x)

evaluates the first `N+1` orthogonal polynomials at point `x`,
where `A`, `B`, and `C` are `AbstractVector`s containing the first form recurrence coefficients as defined in 
[DLMF](https://dlmf.nist.gov/18.9) and `N` is the minimum length of `A`, `B`, and `C`,
i.e. it returns Pᵢ(x) for i = 0, 1, ..., `min(length(A), length(B), length(C))`
"""
forwardrecurrence(A::AbstractVector, B::AbstractVector, C::AbstractVector, x...) = forwardrecurrence(min(length(A), length(B), length(C))+1, A, B, C, x...)


