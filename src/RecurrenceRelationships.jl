module RecurrenceRelationships
export forwardrecurrence, forwardrecurrence!, clenshaw, clenshaw!, olver, olver!

# choose the type correctly for polynomials in a variable
polynomialtype(::Type{T}) where T = typeof(zero(T)^2+1)
polynomialtype(::Type{N}) where N<:Number = N
polynomialtype(a::Type...) = promote_type(map(polynomialtype, a)...)


include("forward.jl")
include("clenshaw.jl")
include("olver.jl")

end # module RecurrenceRelationships
