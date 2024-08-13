module RecurrenceRelationships
using LinearAlgebra
export forwardrecurrence, forwardrecurrence!, clenshaw, clenshaw!

include("forward.jl")
include("clenshaw.jl")

end # module RecurrenceRelationships
