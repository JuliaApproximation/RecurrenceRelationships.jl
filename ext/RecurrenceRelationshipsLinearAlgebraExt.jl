module RecurrenceRelationshipsLinearAlgebraExt
using RecurrenceRelationships, LinearAlgebra

import RecurrenceRelationships: olver

olver(T::Tridiagonal, f, n...; kwds...) = olver(T.dl, T.d, T.du, f, n...; kwds...)
olver(T::SymTridiagonal, f, n...; kwds...) = olver(T.ev, T.dv, T.ev, f, n...; kwds...)

end