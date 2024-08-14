# RecurrenceRelationships.jl
A Julia package for solving three-term recurrence relationships

This package implements simple algorithms for computing solutions to recurrence relationships,
including forward recurrence for initial value-problems, [Olver's algorithm](https://dlmf.nist.gov/3.6#v) for two-point boundary
value problems, and [Clenshaw's algorithm](https://en.wikipedia.org/wiki/Clenshaw_algorithm) for computing dot products with a given vector as needed for
evaluating expansions in orthogonal polynomials.

## 1. Forward recurrence

As an example, consider computing the first million Chebyshev U polynomials:

```julia
julia> using RecurrenceRelationships

julia> n = 1_000_000; x = 0.1;

julia> @time forwardrecurrence(fill(2, n), zeros(n), ones(n), x)
  0.006259 seconds (8 allocations: 30.518 MiB, 19.45% gc time)
1000000-element Vector{Float64}:
  1.0
  0.2
 -0.96
 -0.392
  0.8815999999999999
  0.56832
 -0.767936
 -0.7219072000000001
  ⋮
  0.2777381949694639
 -0.9332843889921485
 -0.4643950727678936
  0.8404053744385698
  0.6324761476556076
 -0.7139101449074483
 -0.7752581766370972
```
Note this is faster than the explicit formula:
```julia
ulia> @time sin.((1:n) .* acos(x)) ./ sin(acos(x))
  0.010828 seconds (9 allocations: 7.630 MiB)
1000000-element Vector{Float64}:
  1.0
  0.2
 -0.96
 -0.392
  0.8816
  0.5683199999999999
 -0.7679359999999998
 -0.7219072000000001
  ⋮
  0.27773819504129643
 -0.9332843889430753
 -0.46439507272938874
  0.840405374430785
  0.6324761477118992
 -0.713910144815891
 -0.7752581766080119
```
Note forward recurrence is actually more accurate than the explicit formula which can be seen by comparison with a high precision calculation:
```julia
julia> norm(forwardrecurrence(fill(2, n), zeros(n), ones(n), x) - forwardrecurrence(fill(2, n), zeros(n), ones(n), big(x)))
5.93151258729473921191879934738972139533978757730288453476821870826190721098765e-10

julia> norm(sin.((1:n) .* acos(x)) ./ sin(acos(x)) - forwardrecurrence(fill(2, n), zeros(n), ones(n), big(x)))
4.538171757754684777956652395339636096999624380286573911589424226541646390097931e-08
```

We can make it even faster using FillArrays.jl:
```julia

```
