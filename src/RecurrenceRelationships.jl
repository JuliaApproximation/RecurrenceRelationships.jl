module RecurrenceRelationships
export forwardrecurrence, forwardrecurrence!, clenshaw, clenshaw!, olver, olver!

# choose the type correctly for polynomials in a variable
polynomialtype(::Type{C}, ::Type{X}) where {C,X} = typeof(zero(C)*zero(X)^2+one(C)*one(X))
polynomialtype(a::Type{<:Number}, b::Type{N}) where N<:AbstractMatrix{<:Number} = N # TODO: use EltypeExtensions.jl
polynomialtype(a::Type) = polynomialtype(a, a)


include("forward.jl")
include("clenshaw.jl")
include("olver.jl")

end # module RecurrenceRelationships
