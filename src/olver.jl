

Base.@propagate_inbounds @inline function olver_forward_next(d, r, a, b, c, f, k)
    r̃ₖ = inv(r[k-1])
    f[k] - a[k-1]d[k-1]r̃ₖ, b[k] - c[k-1]a[k-1]r̃ₖ
end

Base.@propagate_inbounds @inline olver_backward_next(d, r, c, k) = (d[k]-c[k]*d[k+1])/r[k]


function olver_forward!(d::AbstractVector{T}, r, a, b, c, f, N; atol=eps(T)) where T
    Ñ = min(2N+10,length(f))
    resize!(r, Ñ)
    resize!(d, Ñ)

    r[1] = b[1]
    M = zero(T)
    # d[1] is left unmodified
    d[1] = f[1]
    for k = 2:N
        d[k],r[k] = olver_forward_next(d, r, a, b, c, f, k)
        M = max(M*abs(c[k-1]),one(T)) / abs(r[k])
    end

    for k = N+1:length(f)
        if k > length(r)
            Ñ = min(2length(r),length(f))
            resize!(r, Ñ)
            resize!(d, Ñ)
        end
        d[k],r[k] = olver_forward_next(d, r, a, b, c, f, k)
        M = M*abs(c[k-1]) / abs(r[k])
        ε = M * abs(d[k])
        if ε ≤ atol
            return resize!(d,k-1), resize!(r,k), ε
        end
    end
    d, r, zero(T)
end

function olver_forward!(d::AbstractVector{T}, r, a, b, c, f; atol=eps(T)) where T
    N = length(d)
    Ñ = min(2N+10,length(f))
    resize!(r, Ñ)
    resize!(d, Ñ)

    r[1] = b[1]
    M = zero(T)
    # d[1] is left unmodified
    d[1] = f[1]

    for k = 2:length(f)
        if k > length(r)
            Ñ = min(2length(r),length(f))
            resize!(r, Ñ)
            resize!(d, Ñ)
        end
        d[k],r[k] = olver_forward_next(d, r, a, b, c, f, k)
        M = max(M*abs(c[k-1]),1) / abs(r[k])
        ε = M * abs(d[k])
        if ε ≤ atol
            return resize!(d,k-1), resize!(r,k), ε
        end
    end
    d, r, zero(T)
end

function olver!(d::AbstractVector{T}, r, a, b, c, f; atol=eps(T)) where T
    olver_forward!(d, r, a, b, c, f; atol)
    # backsubstitution
    M = length(d)
    d[M] /= r[M]
    for k = M-1:-1:1
        d[k] = olver_backward_next(d, r, c, k)
    end
    d
end


function olver!(d::AbstractVector{T}, r, a, b, c, f, N; atol=eps(T)) where T
    olver_forward!(d, r, a, b, c, f, N...; atol)
    # backsubstitution
    M = length(d)
    d[M] /= r[M]
    for k = M-1:-1:1
        d[k] = olver_backward_next(d, r, c, k)
    end
    resize!(d, N)
end


"""
    olver(a, b, c, f, N; atol)

returns a vector `u` satisfying the 3-term recurrence relationship

    -b[1]u[1] + c[1]u[2] = f[1]
    a[k-1]u[k-1] - b[k]u[k] + c[k]u[k+1] = f[k]

It will compute at least `N` entries, but possibly more, returning the result
when the backward error of the first `N` entries between consective truncations is less than `atol`.
"""
function olver(a, b, c, f::AbstractVector{T}, N; atol=eps(float(T))) where T
    dest = Vector{float(T)}(undef, N)
    olver!(dest, similar(dest), a, b, c, f, N; atol)
end

"""
    olver(a, b, c, f; atol)

returns a vector `u` satisfying the 3-term recurrence relationship

    -b[1]u[1] + c[1]u[2] = f[1]
    a[k-1]u[k-1] - b[k]u[k] + c[k]u[k+1] = f[k]

It will compute the result
when the backward error between consective truncations is less than `atol`.
"""
function olver(a, b, c, f::AbstractVector{T}; atol=eps(float(T))) where T
    dest = Vector{float(T)}(undef, 1)
    olver!(dest, similar(dest), a, b, c, f; atol)
end