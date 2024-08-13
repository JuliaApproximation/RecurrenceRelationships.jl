module RecurrenceRelationships
using LinearAlgebra
export forwardrecurrence, clenshaw

include("forward.jl")
include("clenshaw.jl")

end # module RecurrenceRelationships
