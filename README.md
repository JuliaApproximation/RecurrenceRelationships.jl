# RecurrenceRelationships.jl
A Julia package for solving three-term recurrence relationships

This package implements simple algorithms for computing solutions to recurrence relationships,
including forward recurrence for initial value-problems, [Olver's algorithm](https://dlmf.nist.gov/3.6#v) for two-point boundary
value problems, and [Clenshaw's algorithm](https://en.wikipedia.org/wiki/Clenshaw_algorithm) for computing dot products with a given vector as needed for
evaluating expansions in orthogonal polynomials.

## Forward recurrence

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
julia> θ = acos(x); @time sin.((1:n) .* θ) ./ sin(θ)
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
Note forward recurrence is actually more accurate than the explicit formula which can be seen by comparison with a high precision calculation (though the accuracy is worse near ±1):
```julia
julia> norm(forwardrecurrence(fill(2, n), zeros(n), ones(n), x) - forwardrecurrence(fill(2, n), zeros(n), ones(n), big(x)))
5.93151258729473921191879934738972139533978757730288453476821870826190721098765e-10

julia> norm(sin.((1:n) .* θ) ./ sin(θ) - forwardrecurrence(fill(2, n), zeros(n), ones(n), big(x)))
4.538171757754684777956652395339636096999624380286573911589424226541646390097931e-08
```

We can make it even faster using FillArrays.jl:
```julia
julia> using FillArrays

julia> @time forwardrecurrence(Fill(2, n), Zeros(n), Ones(n), x)
  0.003387 seconds (5 allocations: 7.630 MiB)
1000000-element Vector{Float64}:
  1.0
  0.2
 -0.96
  ⋮
  0.6324761476556076
 -0.7139101449074483
 -0.7752581766370972
```

We can also use LazyArrays.jl to represent Chebyshev T recurrence lazily:
```julia
julia> using LazyArrays

julia> @time forwardrecurrence(Vcat(1, Fill(2, n-1)), Zeros(n), Ones(n), x)
  0.002740 seconds (103 allocations: 7.634 MiB)
1000000-element Vector{Float64}:
  1.0
  0.1
 -0.98
 -0.296
  0.9208
  0.48016
 -0.824768
 -0.6451136
  0.6957452799999999
  ⋮
  0.9968292069233
  0.17885499217086823
 -0.9610582084891264
 -0.3710666338686935
  0.8868448817153877
  0.548435610211771
 -0.7771577596730335
 -0.7038671621463777
 ```
 And this matches the explicit formula:
 ```julia
julia> @time cos.((0:n-1) .* θ)
  0.042121 seconds (6 allocations: 7.630 MiB, 75.68% gc time)
1000000-element Vector{Float64}:
  1.0
  0.1
 -0.98
 -0.29600000000000004
  0.9208
  0.48016000000000003
 -0.824768
 -0.6451136000000001
  0.6957452799999999
  ⋮
  0.9968292069266631
  0.178854992187329
 -0.9610582084686914
 -0.37106663377504573
  0.8868448817354809
  0.5484356102238197
 -0.7771577596278382
 -0.7038671620731024
 ```



## Olver's algorithm

## Clenshaw's algorithm

Clenshaw's algorithm is an efficient way to compute expansions in orthogonal polynomials. Here we compute
$$
∑_{k=0}^{n-1} {U_k(x) \over k+1}
$$
for $x = 0.1$ and $n = 1,000,000$:

```julia
julia> @time clenshaw(inv.(1:n), fill(2, n), zeros(n), ones(n+1), x)
  0.006446 seconds (12 allocations: 30.518 MiB)
0.8396901361362448
```

This matches the explicit expression:
```julia
julia> @time sum(sin((k+1)*θ)/(k+1) for k=0:n-1)/sin(θ)
  0.161225 seconds (8.03 M allocations: 124.408 MiB, 2.74% gc time, 16.44% compilation time)
0.8396901361362544
```

Again, using FillArrays.jl is faster. And we can use InfiniteArrays.jl to allow any length of
coefficients:
```julia
julia> using InfiniteArrays

julia> @time clenshaw(inv.(1:n), Fill(2, ∞), Zeros(∞), Ones(∞), x)
  0.004574 seconds (5 allocations: 7.630 MiB)
0.8396901361362448
```