using RecurrenceRelationships, LinearAlgebra, Test
using FillArrays, LazyArrays
using DynamicPolynomials


@testset "forward" begin
    @testset "Chebyshev U" begin
        N = 5
        A, B, C = Fill(2,N-1), Zeros{Int}(N-1), Ones{Int}(N)
        @testset "forwardrecurrence!" begin
            @test @inferred(forwardrecurrence(N, A, B, C, 1)) == @inferred(forwardrecurrence!(Vector{Int}(undef,N), A, B, C, 1)) == 1:N
            @test forwardrecurrence!(Vector{Int}(undef,N), A, B, C, -1) == (-1) .^ (0:N-1) .* (1:N)
            @test forwardrecurrence(N, A, B, C, 0.1) ≈ forwardrecurrence!(Vector{Float64}(undef,N), A, B, C, 0.1) ≈
                    sin.((1:N) .* acos(0.1)) ./ sqrt(1-0.1^2)
        end
    end

    @testset "Chebyshev-as-general" begin
        N = 5
        A, B, C = [1; fill(2,N-2)], fill(0,N-1), fill(1,N)
        Af, Bf, Cf = float(A), float(B), float(C)
        @test forwardrecurrence(N, A, B, C, 1) == forwardrecurrence!(Vector{Int}(undef,N), A, B, C, 1) == ones(Int,N)
        @test forwardrecurrence!(Vector{Int}(undef,N), A, B, C, -1) == (-1) .^ (0:N-1)
        @test forwardrecurrence(N, A, B, C, 0.1) ≈ forwardrecurrence!(Vector{Float64}(undef,N), A, B, C, 0.1) ≈ cos.((0:N-1) .* acos(0.1))
    end

    @testset "Legendre" begin
        @testset "Float64" begin
            N = 5
            n = 0:N-1
            A = (2n .+ 1) ./ (n .+ 1)
            B = zeros(N)
            C = n ./ (n .+ 1)
            v_1 = forwardrecurrence(N, A, B, C, 1)
            v_f = forwardrecurrence(N, A, B, C, 0.1)
            @test v_1 ≈ ones(N)
            @test forwardrecurrence(N, A, B, C, -1) ≈ (-1) .^ (0:N-1)
            @test v_f ≈ [1,0.1,-0.485,-0.1475,0.3379375]
        end

        @testset "BigFloat" begin
            N = 5
            n = BigFloat(0):N-1
            A = (2n .+ 1) ./ (n .+ 1)
            B = zeros(N)
            C = n ./ (n .+ 1)
            @test forwardrecurrence(N, A, B, C, parse(BigFloat,"0.1")) ≈ [1,big"0.1",big"-0.485",big"-0.1475",big"0.3379375"]
        end
    end

    @testset "Int" begin
        N = 10; A = 1:10; B = 2:11; C = range(3; step=2, length=N+1)
        v_i = forwardrecurrence(N, A, B, C, 1)
        v_f = forwardrecurrence(N, A, B, C, 0.1)
        @test v_i isa Vector{Int}
        @test v_f isa Vector{Float64}
    end

    @testset "Tridiagonal and forward recurrence" begin
        N = 2000
        z = 100.1
        a,b,c = ones(N-1) .- 0.5*(1:N-1) .^ (-1), -range(2; step=2, length=N)/z, ones(N-1)  .+ (1:N-1) .^ (-2)
        A,B,C = 1 ./ c, -b[1:end-1]./c, [0; a[1:end-1]./c[2:end]]
        x = 0.1
        u_for = forwardrecurrence(10, A, B, C, x)
        @test u_for[2] ≈ (A[1]*x + B[1])
        @test u_for[3] ≈ (A[2]*x + B[2])u_for[2] - C[2]u_for[1]
        u_bac = ([[1; zeros(N-1)]'; Tridiagonal(a, Vector(b) .- x, c)[1:end-1,:]] \ [1; zeros(N-1)])[1:10]
        @test u_for ≈ u_bac
    end
end
@testset "clenshaw" begin
    @testset "Chebyshev T" begin
        c = [1,2,3]
        @test @inferred(clenshaw(c,1)) ≡ 1 + 2 + 3
        @test @inferred(clenshaw(c,0)) ≡ 1 + 0 - 3
        @test @inferred(clenshaw(c,[-1,0,1])) == clenshaw!([-1,0,1],c) == [2,-2,6]
        @test @inferred(clenshaw(c,0.1)) == 1 + 2*0.1 + 3*cos(2acos(0.1))
        @test clenshaw(c,[-1,0,1]) isa Vector{Int}

        @test clenshaw(Int[], 0) ≡ 0
        @test clenshaw([1], 0) ≡ 1
        @test clenshaw([1,2], 0) ≡ 1

        for elty in (Float64, Float32)
            cf = elty.(c)
            @test @inferred(clenshaw(elty[],1)) ≡ zero(elty)

            x = elty[1,0,0.1]
            @test @inferred(clenshaw(c,x)) ≈ @inferred(clenshaw!(copy(x),c)) ≈
                @inferred(clenshaw!(similar(x),c,x)) ≈
                @inferred(clenshaw(cf,x)) ≈ @inferred(clenshaw!(copy(x),cf)) ≈
                @inferred(clenshaw!(similar(x),cf,x)) ≈ elty[6,-2,-1.74]

            @testset "Strided" begin
                cv = view(cf,:)
                xv = view(x,:)
                @test clenshaw!(similar(xv), cv, xv) == clenshaw!(similar(x), cf, x)

                cv2 = view(cf,1:2:3)
                @test clenshaw!(similar(xv), cv2, xv) == clenshaw([1,3], x)

                # modifies x and xv
                @test clenshaw!(xv, cv2) == xv == x == clenshaw([1,3], elty[1,0,0.1])
            end
        end
        @testset "matrix coefficients" begin
            c = [1 2; 3 4; 5 6]
            @test clenshaw(c,0.1; dims=1) ≈ [clenshaw(c[:,1],0.1), clenshaw(c[:,2],0.1)]'
            @test clenshaw(c,0.1; dims=2) ≈ [clenshaw(c[1,:],0.1); clenshaw(c[2,:],0.1); clenshaw(c[3,:],0.1) ;;]
        end
    end

    @testset "Chebyshev U" begin
        N = 5
        A, B, C = Fill(2,N-1), Zeros{Int}(N-1), Ones{Int}(N)
        c = [1,2,3]
        @test c'forwardrecurrence(3, A, B, C, 0.1) ≈ clenshaw([1,2,3], A, B, C, 0.1) ≈
            1 + (2sin(2acos(0.1)) + 3sin(3acos(0.1)))/sqrt(1-0.1^2)

        @testset "matrix coefficients" begin
            c = [1 2; 3 4; 5 6]
            @test clenshaw(c,A,B,C,0.1; dims=1) ≈ [clenshaw(c[:,1],A,B,C,0.1), clenshaw(c[:,2],A,B,C,0.1)]'
            @test clenshaw(c,A,B,C,0.1; dims=2) ≈ [clenshaw(c[1,:],A,B,C,0.1); clenshaw(c[2,:],A,B,C,0.1); clenshaw(c[3,:],A,B,C,0.1) ;;]
        end
    end

    @testset "Chebyshev-as-general" begin
        c, A, B, C = [1,2,3], [1,2,2], fill(0,3), fill(1,4)
        cf, Af, Bf, Cf = float(c), float(A), float(B), float(C)
        @test @inferred(clenshaw(c, A, B, C, 1)) ≡ 6
        @test @inferred(clenshaw(c, A, B, C, 0.1)) ≡ -1.74
        @test @inferred(clenshaw([1,2,3], A, B, C, [-1,0,1])) == clenshaw!([-1,0,1], [1,2,3],A, B, C) == [2,-2,6]
        @test clenshaw(c, A, B, C, [-1,0,1]) isa Vector{Int}
        @test @inferred(clenshaw(Float64[], A, B, C, 1)) ≡ 0.0

        x = [1,0,0.1]
        @test @inferred(clenshaw(c, A, B, C, x)) ≈ @inferred(clenshaw!(copy(x), c, A, B, C)) ≈
            @inferred(clenshaw!(similar(x), c, A, B, C, x, one.(x))) ≈
            @inferred(clenshaw!(similar(x), cf, Af, Bf, Cf, x, one.(x))) ≈
            @inferred(clenshaw([1.,2,3], A, B, C, x)) ≈
            @inferred(clenshaw!(copy(x), [1.,2,3], A, B, C)) ≈ [6,-2,-1.74]
    end

    @testset "Legendre" begin
        @testset "Float64" begin
            N = 5
            n = 0:N-1
            A = (2n .+ 1) ./ (n .+ 1)
            B = zeros(N)
            C = n ./ (n .+ 1)
            v_1 = forwardrecurrence(N, A, B, C, 1)
            v_f = forwardrecurrence(N, A, B, C, 0.1)

            n = 0:N # need extra entry for C in Clenshaw
            C = n ./ (n .+ 1)
            for j = 1:N
                c = [zeros(j-1); 1]
                @test clenshaw(c, A, B, C, 1) ≈ v_1[j] # Julia code
                @test clenshaw(c, A, B, C, 0.1) ≈  v_f[j] # Julia code
                @test clenshaw!([0.0,0.0], c, A, B, C, [1.0,0.1], [1.0,1.0])  ≈ [v_1[j],v_f[j]] # libfasttransforms
            end
        end
    end

    @testset "Int" begin
        N = 10; A = 1:10; B = 2:11; C = range(3; step=2, length=N+1)
        v_i = forwardrecurrence(N, A, B, C, 1)

        j = 3
        @test clenshaw([zeros(Int,j-1); 1; zeros(Int,N-j)], A, B, C, 1) == v_i[j]
    end

    @testset "Zeros diagonal" begin
        N = 10; A = randn(N); B = Zeros{Int}(N); C = randn(N+1)
        @test forwardrecurrence(N, A, B, C, 0.1) == forwardrecurrence(N, A, Vector(B), C, 0.1)
        c = randn(N)
        @test clenshaw(c, A, B, C, 0.1) == clenshaw(c, A, Vector(B), C, 0.1)
    end

    @testset "LazyArrays" begin
        n = 1000
        x = 0.1
        θ = acos(x)
        @test forwardrecurrence(Vcat(1, Fill(2, n-1)), Zeros(n), Ones(n), x) ≈ cos.((0:n-1) .* θ)
        @test clenshaw((1:n), Vcat(1, Fill(2, n-1)), Zeros(n), Ones(n+1), x) ≈ sum(cos(k * θ) * (k+1) for k = 0:n-1)
    end

    @testset "2D" begin
        # cheb U recurrence
        m,n = 5,6
        coeffs = ((1:m) .+ 2(1:n)')
        x,y = 0.1,0.2
        A, B, C = Fill(2,n), Zeros{Int}(n), Ones{Int}(n+1)
        

        A_T, B_T, C_T = [1; fill(2,m-1)], fill(0,m), fill(1,m+1)
        @test clenshaw(vec(clenshaw(coeffs, x; dims=1)), A, B, C, y) ≈ clenshaw(vec(clenshaw(coeffs, A, B, C, y; dims=2)), x) ≈
                only(clenshaw!([0.0], clenshaw!(Matrix{Float64}(undef,1,n), coeffs, x), A, B, C, y)) ≈ 
                only(clenshaw!([0.0], clenshaw!(Matrix{Float64}(undef,m,1), coeffs, A, B, C, y), x)) ≈
                forwardrecurrence(A_T, B_T, C_T, x)'coeffs*forwardrecurrence(A, B, C, y)
    end
end

@testset "olver" begin
    @testset "Bessel" begin
        N = 1000
        x = 0.1
        a,b,c = ones(N-1), -range(2; step=2, length=N)/x, ones(N-1)
        f = [1; zeros(N-1)]
        j = olver(a, b, c, f)
        @test j == olver(SymTridiagonal(Vector(b), c), f) == olver(Tridiagonal(a, Vector(b), c), f)
        @test j[1:5] ≈ olver(a, b, c, f, 5) ≈ olver(a, b, c, f, 100)[1:5]
        @test length(olver(a, b, c, f, 100)) == 100
        @test olver(a, b, c, [1]) ≈ [-0.05]

        T = SymTridiagonal(Vector(b), c)
        L, U = lu(Matrix(T), NoPivot()) # Matrix due to v1.6 not supporting SymTridiagonal
        n = length(j)
        @test U[1:n,1:n] \ (L[1:n,1:n] \ [1; zeros(n-1)]) ≈ j
    end

    @testset "non-symmetric" begin
        N = 2000
        z = 30.1
        a,b,c = 2ones(N-1) .- 0.5*cos.(1:N-1), -range(2; step=2, length=N)/z, 2ones(N-1)  .+ sin.(1:N-1)

        f = [cos.(-(1:50)); exp.(-(1:N-50))]
        u = olver(a, b, c, f)
        T = Tridiagonal(a, Vector(b), c)
        @test u == olver(T, f)
        L, U = lu(Matrix(T), NoPivot()) # Matrix due to v1.6 not supporting SymTridiagonal
        n = length(u)

        @testset "error" begin
            d = [0.0]; r = [0.0];
            d,r,ε = RecurrenceRelationships.olver_forward!(d, r, a, b, c, f; atol=0.1)
            n = length(d)
            @test Bidiagonal(ones(n+1), a[1:n] ./ r[1:n], :L) ≈ L[1:n+1,1:n+1]

            g = L[1:n+1,1:n+1] \ f[1:n+1]
            @test g[1:n] ≈ d
            er = U[1:n+1,1:n+1] \ ([d; 0] -  g)
            @test !(maximum(abs,er) ≈ last(er)) # make sure non-degenerate error
            A,B,C = 1 ./ c, -b[1:end-1]./c, [0; a[1:end-1]./c[2:end]]
            p = forwardrecurrence(n+2, A, B, C)
            @test norm(T[1:n+1,1:n+2]*p) ≤ 10E-14
            @test g[n+1]*p[1:n+1] ./ (p[n+2]c[n+1]) ≈ er

            @test c[1]p[2]/p[1] ≈ -r[1]
            @test c[2]p[3]/p[2] ≈ -r[2]
            @test p[n+1]/(p[n+2] * c[n+1]) ≈ -1/r[n+1]

            @test maximum(abs, er) ≈ ε # we have captured the exact error
        end

        @testset "finite error" begin
            for k = 1:10
                d = [0.0]; r = [0.0];
                d,r,ε = RecurrenceRelationships.olver_forward!(d, r, a, b, c, f,k; atol=0.1)
                n = length(d)

                g = L[1:n+1,1:n+1] \ f[1:n+1]
                @test g[1:n] ≈ d
                er = U[1:n+1,1:n+1] \ ([d; 0] -  g)

                @test maximum(abs, er[1:k]) ≈ ε # we have captured the exact error
            end

            d = [0.0]; r = [0.0];
            d,r,ε = RecurrenceRelationships.olver_forward!(d, r, a, b, c, f,N; atol=0.1)
            @test iszero(ε)
            @test length(olver(a, b, c, f, N)) == N
        end
    end

    @testset "complex" begin
        N = 100
        a,b,c = Fill(1/2,N-1), Zeros{Int}(N), Fill(1/2,N-1)
        z = 1 + im
        f = [-π/2; zeros(N-1)]
        u = olver(a, b .- z, c, f)
        ζ = z - sqrt(z-1)*sqrt(z+1)
        @test π*ζ .^ (1:length(u)) ≈ u
        @test u ≈ olver(a, b .- z, c, f .+ 0im)
        @test u[1:5] ≈ olver(a, b .- z, c, f .+ 0im, 5)
    end
end

@testset "DynamicPolynomials" begin
    @polyvar x
    N = 5
    A, B, C = Fill(2,N-1), Zeros{Int}(N-1), Ones{Int}(N)
    @test @inferred(forwardrecurrence(N, A, B, C, x)) == [1,2x,4x^2-1, 8x^3-4x, 16x^4 - 12x^2 + 1]
end