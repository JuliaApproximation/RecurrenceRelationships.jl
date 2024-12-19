module RecurrenceRelationships
export forwardrecurrence, forwardrecurrence!, clenshaw, clenshaw!, olver, olver!

# choose the type correctly for polynomials in a variable
polynomialtype(::Type{T}) where T = typeof(zero(T)^2+1)
polynomialtype(::Type{N}) where N<:Number = N
polynomialtype(::Type{N}) where N<:AbstractMatrix{<:Number} = N
polynomialtype(a::Type, b::Type) = promote_type(polynomialtype(a), polynomialtype(b))
polynomialtype(a::Type, b::Type, c::Type...) = polynomialtype(polynomialtype(a, b), c...)
polynomialtype(a::Type{<:Number}, b::Type{N}) where N<:AbstractMatrix{<:Number} = N


include("forward.jl")
include("clenshaw.jl")
include("olver.jl")

end # module RecurrenceRelationships
