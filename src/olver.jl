function olver!(d::AbstractVector{T}, r, a, b, c; atol=eps(T)) where T
    resize!(r, max(length(r), 10))

    r[1] = b[1]/c[1]
    M = zero(T)
    # d[1] is left unmodified
    for k = 2:length(d)
        r[k] = b[k] - c[k-1]a[k-1]/r[k-1]
        d[k] -= a[k-1]d[k-1]/r[k-1]
        M = max(M,one(T)) / abs(r[k])
        if M * abs(d[k]) ≤ atol
            resize!(d, k)
            break
        end
    end

    # backsubstitution
    N = length(d)
    d[N] /= -r[N]
    for k = length(d)-1:-1:1
        d[k] = (d[k]-c[k]*d[k+1])/r[k]
    end
    d
end

olver(d::AbstractVector{T}, a, b, c; atol=eps(float(T))) where T = olver!(convert(AbstractVector{float(T)}, Base.copymutable(d)), similar(d), a, b, c; atol)