module RecurrenceRelationships
using LinearAlgebra
export forwardrecurrence, forwardrecurrence!, clenshaw, clenshaw!

include("forward.jl")
include("clenshaw.jl")

if !isdefined(Base, :get_extension)
    include("../ext/RecurrenceRelationshipsFillArraysExt.jl")
    include("../ext/RecurrenceRelationshipsLazyArraysExt.jl")
end


end # module RecurrenceRelationships
