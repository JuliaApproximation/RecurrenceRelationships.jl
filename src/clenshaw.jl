

"""
clenshaw!(c, A, B, C, x)

evaluates the orthogonal polynomial expansion with coefficients `c` at points `x`,
where `A`, `B`, and `C` are `AbstractVector`s containing the recurrence coefficients
as defined in DLMF,
overwriting `x` with the results.

If `c` is a matrix this treats each column as a separate vector of coefficients, returning a vector
if `x` is a number and a matrix if `x` is a vector.
"""
clenshaw!(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::AbstractVector) =
    clenshaw!(c, A, B, C, x, Ones{eltype(x)}(length(x)), x)

clenshaw!(c::AbstractMatrix, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::Number, f::AbstractVector) =
    clenshaw!(c, A, B, C, x, one(eltype(x)), f)


clenshaw!(c::AbstractMatrix, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::AbstractVector, f::AbstractMatrix) =
    clenshaw!(c, A, B, C, x, Ones{eltype(x)}(length(x)), f)


"""
clenshaw!(c, A, B, C, x, ϕ₀, f)

evaluates the orthogonal polynomial expansion with coefficients `c` at points `x`,
where `A`, `B`, and `C` are `AbstractVector`s containing the recurrence coefficients
as defined in DLMF and ϕ₀ is the zeroth polynomial,
overwriting `f` with the results.
"""
function clenshaw!(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::AbstractVector, ϕ₀::AbstractVector, f::AbstractVector)
    f .= ϕ₀ .* clenshaw.(Ref(c), Ref(A), Ref(B), Ref(C), x)
end


function clenshaw!(c::AbstractMatrix, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::Number, ϕ₀::Number, f::AbstractVector)
    size(c,2) == length(f) || throw(DimensionMismatch("coeffients size and output length must match"))
    @inbounds for j in axes(c,2)
        f[j] = ϕ₀ * clenshaw(view(c,:,j), A, B, C, x)
    end
    f
end

function clenshaw!(c::AbstractMatrix, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::AbstractVector, ϕ₀::AbstractVector, f::AbstractMatrix)
    (size(x,1),size(c,2)) == size(f) || throw(DimensionMismatch("coeffients size and output length must match"))
    @inbounds for j in axes(c,2)
        clenshaw!(view(c,:,j), A, B, C, x, ϕ₀, view(f,:,j))
    end
    f
end

Base.@propagate_inbounds _clenshaw_next(n, A, B, C, x, c, bn1, bn2) = muladd(muladd(A[n],x,B[n]), bn1, muladd(-C[n+1],bn2,c[n]))
Base.@propagate_inbounds _clenshaw_next(n, A, ::Zeros, C, x, c, bn1, bn2) = muladd(A[n]*x, bn1, muladd(-C[n+1],bn2,c[n]))
# Chebyshev U
Base.@propagate_inbounds _clenshaw_next(n, A::AbstractFill, ::Zeros, C::Ones, x, c, bn1, bn2) = muladd(getindex_value(A)*x, bn1, -bn2+c[n])

# allow special casing first arg, for ChebyshevT in OrthogonalPolynomialsQuasi
Base.@propagate_inbounds _clenshaw_first(A, B, C, x, c, bn1, bn2) = muladd(muladd(A[1],x,B[1]), bn1, muladd(-C[2],bn2,c[1]))


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
            bn1,bn2 = _clenshaw_next(n, A, B, C, x, c, bn1, bn2),bn1
        end
        bn1 = _clenshaw_first(A, B, C, x, c, bn1, bn2)
    end
    bn1
end


clenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::AbstractVector) =
    clenshaw!(c, A, B, C, copy(x))

function clenshaw(c::AbstractMatrix, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::Number)
    T = promote_type(eltype(c),eltype(A),eltype(B),eltype(C),typeof(x))
    clenshaw!(c, A, B, C, x, Vector{T}(undef, size(c,2)))
end

function clenshaw(c::AbstractMatrix, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::AbstractVector)
    T = promote_type(eltype(c),eltype(A),eltype(B),eltype(C),typeof(x))
    clenshaw!(c, A, B, C, x, Matrix{T}(undef, size(x,1), size(c,2)))
end

###
# Chebyshev T special cases
###

"""
   clenshaw!(c, x)

evaluates the first-kind Chebyshev (T) expansion with coefficients `c` at points `x`,
overwriting `x` with the results.
"""
clenshaw!(c::AbstractVector, x::AbstractVector) = clenshaw!(c, x, x)


"""
   clenshaw!(c, x, f)

evaluates the first-kind Chebyshev (T) expansion with coefficients `c` at points `x`,
overwriting `f` with the results.
"""
function clenshaw!(c::AbstractVector, x::AbstractVector, f::AbstractVector)
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

function clenshaw!(c::AbstractMatrix, x::Number, f::AbstractVector)
    size(c,2) == length(f) || throw(DimensionMismatch("coeffients size and output length must match"))
    @inbounds for j in axes(c,2)
        f[j] = clenshaw(view(c,:,j), x)
    end
    f
end

function clenshaw!(c::AbstractMatrix, x::AbstractVector, f::AbstractMatrix)
    (size(x,1),size(c,2)) == size(f) || throw(DimensionMismatch("coeffients size and output length must match"))
    @inbounds for j in axes(c,2)
        clenshaw!(view(c,:,j), x, view(f,:,j))
    end
    f
end

clenshaw(c::AbstractVector, x::AbstractVector) = clenshaw!(c, copy(x))
clenshaw(c::AbstractMatrix, x::Number) = clenshaw!(c, x, Vector{promote_type(eltype(c),typeof(x))}(undef, size(c,2)))
clenshaw(c::AbstractMatrix, x::AbstractVector) = clenshaw!(c, x, Matrix{promote_type(eltype(c),eltype(x))}(undef, size(x,1), size(c,2)))


Base.@propagate_inbounds _clenshaw_next(n, A::Vcat{<:Any,1,<:Tuple{<:Number,<:AbstractFill}}, B::Zeros, C::Ones, x, c, bn1, bn2) = 
    _clenshaw_next(n, A.args[2], B, C, x, c, bn1, bn2)

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

Base.@propagate_inbounds function _clenshaw_next!(n, A::AbstractVector, B::AbstractVector, C::AbstractVector, x::AbstractMatrix, c, bn1::AbstractMatrix{T}, bn2::AbstractMatrix{T}) where T
    # bn2 .= B[n] .* bn1 .- C[n+1] .* bn2
    lmul!(-C[n+1], bn2)
    LinearAlgebra.axpy!(B[n], bn1, bn2)
    muladd!(A[n], x, bn1, one(T), bn2)
    view(bn2,band(0)) .+= c[n]
    bn2
end

# Operator * f Clenshaw
Base.@propagate_inbounds function _clenshaw_next!(n, A::AbstractFill, ::Zeros, C::Ones, X::AbstractMatrix, c, f::AbstractVector, bn1::AbstractVector{T}, bn2::AbstractVector{T}) where T
    muladd!(getindex_value(A), X, bn1, -one(T), bn2)
    bn2 .+= c[n] .* f
    bn2
end

Base.@propagate_inbounds function _clenshaw_next!(n, A, ::Zeros, C, X::AbstractMatrix, c, f::AbstractVector, bn1::AbstractVector{T}, bn2::AbstractVector{T}) where T
    muladd!(A[n], X, bn1, -C[n+1], bn2)
    bn2 .+= c[n] .* f
    bn2
end

Base.@propagate_inbounds function _clenshaw_next!(n, A, B, C, X::AbstractMatrix, c, f::AbstractVector, bn1::AbstractVector{T}, bn2::AbstractVector{T}) where T
    bn2 .= B[n] .* bn1 .- C[n+1] .* bn2 .+ c[n] .* f
    muladd!(A[n], X, bn1, one(T), bn2)
    bn2
end

# allow special casing first arg, for ChebyshevT in ClassicalOrthogonalPolynomials
Base.@propagate_inbounds function _clenshaw_first!(A, ::Zeros, C, X, c, bn1, bn2) 
    muladd!(A[1], X, bn1, -C[2], bn2)
    view(bn2,band(0)) .+= c[1]
    bn2
end

Base.@propagate_inbounds function _clenshaw_first!(A, B, C, X, c, bn1, bn2) 
    lmul!(-C[2], bn2)
    LinearAlgebra.axpy!(B[1], bn1, bn2)
    muladd!(A[1], X, bn1, one(eltype(bn2)), bn2)
    view(bn2,band(0)) .+= c[1]
    bn2
end

Base.@propagate_inbounds function _clenshaw_first!(A, ::Zeros, C, X, c, f::AbstractVector, bn1, bn2) 
    muladd!(A[1], X, bn1, -C[2], bn2)
    bn2 .+= c[1] .* f
    bn2
end

Base.@propagate_inbounds function _clenshaw_first!(A, B, C, X, c, f::AbstractVector, bn1, bn2) 
    bn2 .= B[1] .* bn1 .- C[2] .* bn2 .+ c[1] .* f
    muladd!(A[1], X, bn1, one(eltype(bn2)), bn2)
    bn2
end

_clenshaw_op(::AbstractBandedLayout, Z, N) = BandedMatrix(Z, (N-1,N-1))

function clenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, X::AbstractMatrix)
    N = length(c)
    T = promote_type(eltype(c),eltype(A),eltype(B),eltype(C),eltype(X))
    @boundscheck check_clenshaw_recurrences(N, A, B, C)
    m = size(X,1)
    m == size(X,2) || throw(DimensionMismatch("X must be square"))
    N == 0 && return zero(T)
    bn2 = _clenshaw_op(MemoryLayout(X), Zeros{T}(m, m), N)
    bn1 = _clenshaw_op(MemoryLayout(X), c[N]*Eye{T}(m), N)
    _clenshaw_op!(c, A, B, C, X, bn1, bn2)
end

function clenshaw(c::AbstractVector, A::AbstractVector, B::AbstractVector, C::AbstractVector, X::AbstractMatrix, f::AbstractVector)
    N = length(c)
    T = promote_type(eltype(c),eltype(A),eltype(B),eltype(C),eltype(X))
    @boundscheck check_clenshaw_recurrences(N, A, B, C)
    m = size(X,1)
    m == size(X,2) || throw(DimensionMismatch("X must be square"))
    m == length(f) || throw(DimensionMismatch("Dimensions must match"))
    N == 0 && return [zero(T)]
    bn2 = zeros(T,m)
    bn1 = Vector{T}(undef,m)
    bn1 .= c[N] .* f
    _clenshaw_op!(c, A, B, C, X, f, bn1, bn2)
end

function _clenshaw_op!(c, A, B, C, X, bn1, bn2)
    N = length(c)
    N == 1 && return bn1
    @inbounds begin
        for n = N-1:-1:2
            bn1,bn2 = _clenshaw_next!(n, A, B, C, X, c, bn1, bn2),bn1
        end
        bn1 = _clenshaw_first!(A, B, C, X, c, bn1, bn2)
    end
    bn1
end

function _clenshaw_op!(c, A, B, C, X, f::AbstractVector, bn1, bn2)
    N = length(c)
    N == 1 && return bn1
    @inbounds begin
        for n = N-1:-1:2
            bn1,bn2 = _clenshaw_next!(n, A, B, C, X, c, f, bn1, bn2),bn1
        end
        bn1 = _clenshaw_first!(A, B, C, X, c, f, bn1, bn2)
    end
    bn1
end



"""
    Clenshaw(a, X)

represents the operator `a(X)` where a is a polynomial.
Here `a` is to stored as a quasi-vector.
"""
struct Clenshaw{T, Coefs<:AbstractVector, AA<:AbstractVector, BB<:AbstractVector, CC<:AbstractVector, Jac<:AbstractMatrix} <: AbstractBandedMatrix{T}
    c::Coefs
    A::AA
    B::BB
    C::CC
    X::Jac
    p0::T
end

Clenshaw(c::AbstractVector{T}, A::AbstractVector, B::AbstractVector, C::AbstractVector, X::AbstractMatrix{T}, p0::T) where T = 
    Clenshaw{T,typeof(c),typeof(A),typeof(B),typeof(C),typeof(X)}(c, A, B, C, X, p0)

Clenshaw(c::Number, A, B, C, X, p) = Clenshaw([c], A, B, C, X, p)

function Clenshaw(a::AbstractQuasiVector, X::AbstractQuasiMatrix)
    P,c = arguments(a)
    Clenshaw(paddeddata(c), recurrencecoefficients(P)..., jacobimatrix(X), _p0(P))
end

copy(M::Clenshaw) = M
size(M::Clenshaw) = size(M.X)
axes(M::Clenshaw) = axes(M.X)
bandwidths(M::Clenshaw) = (length(M.c)-1,length(M.c)-1)

Base.array_summary(io::IO, C::Clenshaw{T}, inds::Tuple{Vararg{OneToInf{Int}}}) where T =
    print(io, Base.dims2string(length.(inds)), " Clenshaw{$T} with $(length(C.c)) degree polynomial")

struct ClenshawLayout <: AbstractLazyBandedLayout end
MemoryLayout(::Type{<:Clenshaw}) = ClenshawLayout()
sublayout(::ClenshawLayout, ::Type{<:NTuple{2,AbstractUnitRange{Int}}}) = ClenshawLayout()
sublayout(::ClenshawLayout, ::Type{<:Tuple{AbstractUnitRange{Int},Union{Slice,AbstractInfUnitRange{Int}}}}) = LazyBandedLayout()
sublayout(::ClenshawLayout, ::Type{<:Tuple{Union{Slice,AbstractInfUnitRange{Int}},AbstractUnitRange{Int}}}) = LazyBandedLayout()
sublayout(::ClenshawLayout, ::Type{<:Tuple{Union{Slice,AbstractInfUnitRange{Int}},Union{Slice,AbstractInfUnitRange{Int}}}}) = LazyBandedLayout()
sub_materialize(::ClenshawLayout, V) = BandedMatrix(V)

function _BandedMatrix(::ClenshawLayout, V::SubArray{<:Any,2})
    M = parent(V)
    kr,jr = parentindices(V)
    b = bandwidth(M,1)
    jkr = max(1,min(first(jr),first(kr))-b÷2):max(last(jr),last(kr))+b÷2
    # relationship between jkr and kr, jr
    kr2,jr2 = kr.-first(jkr).+1,jr.-first(jkr).+1
    lmul!(M.p0, clenshaw(M.c, M.A, M.B, M.C, M.X[jkr, jkr])[kr2,jr2])
end

function getindex(M::Clenshaw{T}, kr::AbstractUnitRange, j::Integer) where T
    b = bandwidth(M,1)
    jkr = max(1,min(j,first(kr))-b÷2):max(j,last(kr))+b÷2
    # relationship between jkr and kr, jr
    kr2,j2 = kr.-first(jkr).+1,j-first(jkr)+1
    f = [Zeros{T}(j2-1); one(T); Zeros{T}(length(jkr)-j2)]
    lmul!(M.p0, clenshaw(M.c, M.A, M.B, M.C, M.X[jkr, jkr], f)[kr2])
end

getindex(M::Clenshaw, k::Int, j::Int) = M[k:k,j][1]

function getindex(S::Symmetric{T,<:Clenshaw}, k::Integer, jr::AbstractUnitRange) where T
    m = max(jr.start,jr.stop,k)
    return Symmetric(getindex(S.data,1:m,1:m),Symbol(S.uplo))[k,jr]
end
function getindex(S::Symmetric{T,<:Clenshaw}, kr::AbstractUnitRange, j::Integer) where T
    m = max(kr.start,kr.stop,j)
    return Symmetric(getindex(S.data,1:m,1:m),Symbol(S.uplo))[kr,j]
end
function getindex(S::Symmetric{T,<:Clenshaw}, kr::AbstractUnitRange, jr::AbstractUnitRange) where T
    m = max(kr.start,jr.start,kr.stop,jr.stop)
    return Symmetric(getindex(S.data,1:m,1:m),Symbol(S.uplo))[kr,jr]
end

transposelayout(M::ClenshawLayout) = LazyBandedMatrices.LazyBandedLayout()
# TODO: generalise for layout, use Base.PermutedDimsArray
Base.permutedims(M::Clenshaw{<:Number}) = transpose(M)


function materialize!(M::MatMulVecAdd{<:ClenshawLayout,<:AbstractPaddedLayout,<:AbstractPaddedLayout})
    α,A,x,β,y = M.α,M.A,M.B,M.β,M.C
    length(y) == size(A,1) || throw(DimensionMismatch("Dimensions must match"))
    length(x) == size(A,2) || throw(DimensionMismatch("Dimensions must match"))
    x̃ = paddeddata(x);
    m = length(x̃)
    b = bandwidth(A,1)
    jkr=1:m+b
    p = [x̃; zeros(eltype(x̃),length(jkr)-m)];
    Ax = lmul!(A.p0, clenshaw(A.c, A.A, A.B, A.C, A.X[jkr, jkr], p))
    _fill_lmul!(β,y)
    resizedata!(y, last(jkr))
    v = view(paddeddata(y),jkr)
    LinearAlgebra.axpy!(α, Ax, v)
    y
end




##
# Banded dot is slow
###

LinearAlgebra.dot(x::AbstractVector, A::Clenshaw, y::AbstractVector) = dot(x, mul(A, y))