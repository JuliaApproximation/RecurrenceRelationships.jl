

function olver!(d::AbstractVector{T}, r, a, b, c, f; atol=eps(T)) where T
    N = length(d) # compute at least `N` entries
    resize!(r, N+10)
    resize!(d, N+10)

    r[1] = b[1]/c[1]
    M = zero(T)
    # d[1] is left unmodified
    d[1] = f[1]
    for k = 2:N
        r[k] = b[k] - c[k-1]a[k-1]/r[k-1]
        d[k] = f[k] - a[k-1]d[k-1]/r[k-1]
    end

    for k = N+1:length(f)
        if k > length(r)
            Ñ = min(2length(r),length(f))
            resize!(r, Ñ)
            resize!(d, Ñ)
        end
        r[k] = b[k] - c[k-1]a[k-1]/r[k-1]
        d[k] = f[k] - a[k-1]d[k-1]/r[k-1]
        M = max(M,one(T)) / abs(r[k])
        if M * abs(d[k]) ≤ atol
            resize!(d, k)
            break
        end
    end

    # backsubstitution
    N = length(d)
    d[N] /= -r[N]
    for k = N-1:-1:1
        d[k] = (d[k]-c[k]*d[k+1])/r[k]
    end
    d
end


"""
    olver(a, b, c, f, N=1; atol)

returns a vector `u` satisfying the 3-term recurrence relationship

    -b[1]u[1] + c[1]u[2] = f[1]
    a[k-1]u[k-1] - b[k]u[k] + c[k]u[k+1] = f[k]

It will compute at least `N` entries, but possibly more, returning the result
when the backward error between consective truncations is less than `atol`.
"""
function olver(a, b, c, f::AbstractVector{T}, N=1; atol=eps(float(T))) where T
    dest = Vector{float(T)}(undef, N)
    olver!(dest, similar(dest), a, b, c, f; atol)
end