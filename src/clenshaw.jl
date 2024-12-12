

"""
clenshaw!(f::AbstractVecOrMat, c::AbstractVecOrMat, A, B, C, x::Number)

evaluates the orthogonal polynomial expansion with coefficients `c` at points `x`,
where `A`, `B`, and `C` are `AbstractVector`s containing the recurrence coefficients
as defined in DLMF,
overwriting `f` with the results.

If `c` is a matrix this treats each row (if `size(f,2) == 1`) or column (if `size(f,1) == 1`)` as a separate vector of coefficients.
"""
clenshaw!(f::AbstractVecOrMat, c::AbstractVecOrMat, A::AbstractVector, B::AbstractVector, C::AbstractVector, x) =
    clenshaw!(f, c, A, B, C, x, one(eltype(x)))

clenshaw!(x::AbstractVecOrMat, c::AbstractVecOrMat, A::AbstractVector, B::AbstractVector, C::AbstractVector) =
    clenshaw!(x, c, A, B, C, x)

"""
clenshaw!(f, c, A, B, C, x, ϕ₀)

evaluates the orthogonal polynomial expansion with coefficients `c` at point `x`,
where `A`, `B`, and `C` are `AbstractVector`s containing the recurrence coefficients
as defined in DLMF and ϕ₀ is the zeroth polynomial,
overwriting `f` with the results.
"""
function clenshaw!(f::AbstractVector, c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::AbstractVector, ϕ₀)
    f .= ϕ₀ .* clenshaw.(Ref(c), Ref(A), Ref(B), Ref(C), x)
end

function clenshaw!(f::AbstractMatrix, c::AbstractMatrix, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::Number, ϕ₀::Number)
    m,n = size(c)
    if size(f) == (1,n) # dims = 1
        @inbounds for j in axes(c,2)
            f[1,j] = ϕ₀ * clenshaw(view(c,:,j), A, B, C, x)
        end
    elseif size(f) == (m,1) # dims = 2
        @inbounds for k in axes(c,1)
            f[k,1] = ϕ₀ * clenshaw(view(c,k,:), A, B, C, x)
        end
    else
        throw(DimensionMismatch("coeffients size and output length must match"))
    end
    f
end

Base.@propagate_inbounds clenshaw_next(n, A, B, C, x, c, bn1, bn2) = muladd(muladd(A[n],x,B[n]), bn1, muladd(-C[n+1],bn2,c[n]))

# allow special casing first arg, for ChebyshevT in OrthogonalPolynomialsQuasi
Base.@propagate_inbounds _clenshaw_first(A, B, C, x, c, bn1, bn2) = muladd(muladd(A[1],x,B[1]), bn1, muladd(-C[2],bn2,c[1]))

function check_clenshaw_recurrences(N, A, B, C)
    if length(A) < N || length(B) < N || length(C) < N+1
        throw(ArgumentError("A, B must contain at least $N entries and C must contain at least $(N+1) entries"))
    end
end



"""
    clenshaw(c, A, B, C, x)

evaluates the orthogonal polynomial expansion with coefficients `c` at points `x`,
where `A`, `B`, and `C` are `AbstractVector`s containing the recurrence coefficients
as defined in DLMF.
`x` may also be a single `Number`.

If `c` is a matrix this treats each column as a separate vector of coefficients, returning a vector
if `x` is a number and a matrix if `x` is a vector.
"""
function clenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::Number)
    N = length(c)
    T = promote_type(eltype(c),eltype(A),eltype(B),eltype(C),typeof(x))
    @boundscheck check_clenshaw_recurrences(N, A, B, C)
    N == 0 && return zero(T)
    @inbounds begin
        bn2 = zero(T)
        bn1 = convert(T,c[N])
        N == 1 && return bn1
        for n = N-1:-1:2
            bn1,bn2 = clenshaw_next(n, A, B, C, x, c, bn1, bn2),bn1
        end
        bn1 = _clenshaw_first(A, B, C, x, c, bn1, bn2)
    end
    bn1
end

clenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::AbstractVector) =
    clenshaw!(copy(x), c, A, B, C)

function clenshaw(c::AbstractMatrix, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::Number; dims)
    T = promote_type(eltype(c),eltype(A),eltype(B),eltype(C),typeof(x))
    if dims == 1
        clenshaw!(Matrix{T}(undef, 1, size(c,2)), c, A, B, C, x)
    elseif dims == 2
        clenshaw!(Matrix{T}(undef, size(c,1), 1), c, A, B, C, x)
    else
        throw(ArgumentError("dims must be 1 or 2"))
    end
end

###
# Chebyshev T special cases
###


"""
   clenshaw!(f, c, x)

evaluates the first-kind Chebyshev (T) expansion with coefficients `c` at points `x`,
overwriting `f` with the results.
"""
function clenshaw!(f::AbstractVector, c::AbstractVector, x::AbstractVector)
    f .= clenshaw.(Ref(c), x)
end

"""
    clenshaw(c, x)

evaluates the first-kind Chebyshev (T) expansion with coefficients `c` at  the points `x`.
`x` may also be a single `Number`.
"""
function clenshaw(c::AbstractVector, x::Number)
    N,T = length(c),promote_type(eltype(c),typeof(x))
    if N == 0
        return zero(T)
    elseif N == 1 # avoid issues with NaN x
        return first(c)*one(x)
    end

    y = 2x
    bk1,bk2 = zero(T),zero(T)
    @inbounds begin
        for k = N:-1:2
            bk1,bk2 = muladd(y,bk1,c[k]-bk2),bk1
        end
        muladd(x,bk1,c[1]-bk2)
    end
end

function clenshaw!(f::AbstractVecOrMat, c::AbstractMatrix, x::Number)
    m,n = size(c)
    if size(f) == (1,n) # dims = 1
        @inbounds for j in axes(c,2)
            f[1,j] = clenshaw(view(c,:,j), x)
        end
    elseif size(f) == (m,1) # dims = 2
        @inbounds for k in axes(c,1)
            f[k,1] = clenshaw(view(c,k,:), x)
        end
    else
        throw(DimensionMismatch("coeffients size and output length must match"))
    end
    f
end

clenshaw!(x, c) = clenshaw!(x, c, x)

function clenshaw(c::AbstractMatrix, x::Number; dims)
    T = polynomialtype(eltype(c),typeof(x))
    if dims == 1
        clenshaw!(Matrix{T}(undef, 1, size(c,2)), c, x)
    elseif dims == 2
        clenshaw!(Matrix{T}(undef, size(c,1), 1), c, x)
    else
        throw(ArgumentError("dims must be 1 or 2"))
    end
end


clenshaw(c::AbstractVector, x::AbstractVector) = clenshaw!(copy(x), c)