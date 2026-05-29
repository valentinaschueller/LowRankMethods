export exercise1, exercise2, exercise3, exercise4, exercise5, exercise6, exercise7
export truncated_Tsum
export compare_L

include("transport.jl")
include("rotation.jl")

function stencil_matrix(h, N)::Matrix{Float64}
    return 1 / (2 * h) .* diagm(
        1 => ones(N - 1),
        -1 => -1 * ones(N - 1),
        N - 1 => [-1.0],
        1 - N => [1.0],
    )
end

function exercise1()
    @info "Verify LLRSVD"
    R = rand(30, 20)
    @assert norm(R - todense(LLRSVD(R, 0.0))) < 1e-12
end

function exercise2()
    @info "Verify truncate"
    A = LLRSVD(randn(50, 40), 0.0)
    for tol in [1e-1, 1e-3, 1e-6, 1e-10]
        B = truncate(A, tol)
        @assert norm(todense(A) - todense(B)) < tol
    end

    B = zeros(5, 5)
    M, N = size(B)
    for i in range(1, M)
        for j in range(1, N)
            B[i, j] = 1 / (i + j)
        end
    end
    A = LLRSVD(B, 0.0)
    @info "Singular values of A: $(A.S)"
    TOL = 2e-3
    T = truncate(A, TOL)
    @info "Singular values after truncation with TOL=$TOL: $(T.S)"
    @info "Norm: $(norm(todense(A) - todense(T)))"
end


function exercise3()
    m = 64
    n = 64
    Lx = 2 * π
    Ly = 2 * π
    hx = Lx / m
    hy = Ly / n
    x = LinRange(0, Lx - hx, m)
    y = LinRange(0, Ly - hy, n)
    W0 = sin.(x) * sin.(y')

    Dx = stencil_matrix(hx, m)
    Dy = stencil_matrix(hy, n)

    numdiff = applyL(W0, Dx, Dy)
    truediff = cos.(x) * sin.(y)' + sin.(x) * cos.(y)'
    @info LLRSVD(truediff, 1e-14).S
    @info norm(numdiff - truediff)
end

function exercise4()
    m = 64
    n = 64
    Lx = 2 * π
    Ly = 2 * π
    hx = Lx / m
    hy = Ly / n
    Δt = 0.1
    x = LinRange(0, Lx - hx, m)
    y = LinRange(0, Ly - hy, n)
    W0 = sin.(x) * sin.(y')

    Dx = stencil_matrix(hx, m)
    Dy = stencil_matrix(hy, n)
    Wref = sin.(x .+ Δt) * sin.(y' .+ Δt)

    W1 = applyL(W0, Dx, Dy)
    display(current())
    @info LLRSVD(W1, 1e-10).S

    ks = 0:7
    Ws = zeros(length(ks), m, n)
    Ts = zeros(length(ks), m, n)
    for k in ks
        Ws[k+1, :, :] = W(k, W0, Dx, Dy)
        Ts[k+1, :, :] = taylor_term(k, Δt, W0, Dx, Dy)
    end
    SVDs = [LLRSVD(Ts[k+1, :, :], 0.0) for k in ks]

    plot()
    for k in ks
        plot!(SVDs[k+1].S, label=k)
    end
    plot!(; yscale=:log10, ylabel=L"$σ(T_k)$", xlabel="Index", legend_title=L"k")
    plot!(; title="Singular values of Taylor terms")
    display(current())

    plot(ks, [norm(Ts[k+1, :, :]) for k in ks]; color=:black, m=:xcross)
    plot!(; yscale=:log10, ylabel=L"$\|T_k\|_F$", xlabel=L"k")
    plot!(; title="Frobenius norm of Taylor terms")
    display(current())

    plot(ks, [LLRSVD(Ts[k+1, :, :], 1e-14).r for k in ks]; color=:black, m=:xcross)
    plot!(; xlabel="k", ylabel="Effective rank", title="Rank of Taylor terms", legend=false)
    display(current())

    Tsum = sum(Ts[:, :, :]; dims=1)[1, :, :]
    @info "Effective rank of Taylor series (p=7): $(LLRSVD(Tsum, 1e-14).r)"
    heatmap(Tsum - Wref)
    display(current())
    @info "Difference Taylor series to Wref: $(sqrt(hx * hy) * norm(Tsum - Wref))"

    second_σ = []
    for k in ks
        push!(second_σ, LLRSVD(sum(Ts[1:k+1, :, :]; dims=1)[1, :, :], 0.0).S[2])
    end
    plot(ks, second_σ; color=:black, m=:xcross, label=L"\sigma_2")
    plot!(ks, Δt .^ (ks .- 1); color=:black, m=:xcross, ls=:dash, label=L"\Delta t^{p-1}")
    plot!(; yscale=:log10, ylabel=L"\sigma_2", xlabel=L"p")
    plot!(; title=L"$σ_2(\sum_{k=0}^{p}T_k)$")
    display(current())

    heatmap(W0)
    display(current())

    heatmap(Wref)
    @info "σ of Wref: $(LLRSVD(Wref, 1e-14).S)"
    display(current())

    heatmap(Tsum)
    @info "σ of Tsum: $(LLRSVD(Tsum, 1e-14).S)"
    display(current())
    return
end

function compare_L_and_W()
    m = 64
    n = 64
    Lx = 2 * π
    Ly = 2 * π
    hx = Lx / m
    hy = Ly / n
    x = LinRange(0, Lx - hx, m)
    y = LinRange(0, Ly - hy, n)
    W0 = sin.(x) * sin.(y')
    TOL = 1e-10
    W0L = LLRSVD(W0, TOL)
    p = 8

    Dx = stencil_matrix(hx, m)
    Dy = stencil_matrix(hy, n)

    LW0 = applyL(W0, Dx, Dy)
    LW0L = todense(applyL(W0L, Dx, Dy, 0.0))
    @assert norm(LW0 - LW0L) < 1e-10

    W1 = W(1, W0, Dx, Dy)
    W1L = todense(W(1, W0L, Dx, Dy, 0.0))
    @assert norm(W1 - W1L) < 1e-10

    W3 = W(3, W0, Dx, Dy)
    W3L = todense(W(3, W0L, Dx, Dy, 0.0))
    @assert norm(W3 - W3L) < 1e-10
end

function truncated_Tsum()
    m = 64
    n = 64
    Lx = 2 * π
    Ly = 2 * π
    hx = Lx / m
    hy = Ly / n
    Δt = 0.1
    x = LinRange(0, Lx - hx, m)
    y = LinRange(0, Ly - hy, n)
    W0 = sin.(x) * sin.(y')
    TOL = 1e-10
    W0L = LLRSVD(W0, TOL)
    p = 8

    Dx = stencil_matrix(hx, m)
    Dy = stencil_matrix(hy, n)

    Ts = taylor_step(W0L, Dx, Dy, Δt, p, TOL)
    second_σ = []
    ks = 1:p
    for k in ks
        @info "$k: $(Ts[k+1].S)"
        push!(second_σ, Ts[k+1].S[2])
    end
    plot(ks, second_σ; color=:black, m=:xcross, label=L"\sigma_2")
    plot!(ks, Δt .^ (ks .- 1); color=:black, m=:xcross, ls=:dash, label=L"\Delta t^{p-1}")
    plot!(; yscale=:log10, ylabel=L"\sigma_2", xlabel=L"p")
    plot!(; title=L"$σ_2(\sum_{k=0}^{p}T_k)$")
    display(current())
    # Wref = sin.(x .+ Δt) * sin.(y' .+ Δt)
end

function exercise5()
    @info "Test trunc_sum"
    A = LLRSVD(rand(3, 3), 0.0)
    @info trunc_sum([A; A; A], 1e-4)

    u = [1.0; 0; 0]
    v = [0; 1.0; 0]
    true_sum = u * u' + v * v'
    R1 = LLRSVD(u * u', -1e-16)
    R2 = LLRSVD(v * v', -1e-16)
    truc_sum = trunc_sum([R1; R2], 1e-3)
    @info truc_sum.r
    @info todense(truc_sum)
    @info true_sum
end

function exercise6()
    m = 2^8
    n = 2^8
    Lx = 2 * π
    Ly = 2 * π
    hx = Lx / m
    hy = Ly / n
    Dx = stencil_matrix(hx, m)
    Dy = stencil_matrix(hy, n)
    T = 2 * π
    x = LinRange(0, Lx - hx, m)
    y = LinRange(0, Ly - hy, n)
    TOL = 1e-3
    p = 5
    W0 = sin.(x) * sin.(y')

    W0L = LLRSVD(W0, TOL)
    @info W0L.r

    Δt = min(hx, hy)
    nt = ceil(T / Δt)
    Δt = T / nt
    Wn = time_loop_A(W0L, Dx, Dy, Δt, T, p, TOL)
    @info Wn.r
    heatmap(todense(Wn))
    display(current())

    truc_L = todense(applyL(W0L, Dx, Dy, 1e-14))
    @assert norm(truc_L - W(1, W0, Dx, Dy)) < 1e-10
end

function exercise7()
    m = 128
    n = 128
    Lx = 6
    Ly = 6
    hx = 2 * Lx / m
    hy = 2 * Ly / n
    Dx = stencil_matrix(hx, m)
    Dy = stencil_matrix(hy, n)
    T = 0.5 * Float64(π)
    x = LinRange(-Lx, Lx - hx, m)
    y = LinRange(-Ly, Ly - hy, n)
    X = diagm(x)
    Y = diagm(y)
    TOL = 1e-3
    p = 3

    a = 1.0
    b = 0.5
    x0 = 0.0
    y0 = 0.0
    W0 = zeros(m, n)
    @. W0 = exp(-(x - x0)^2 / (2 * a^2)) * exp(-(y - y0)^2 / (2 * b^2))'
    W0L = LLRSVD(W0, TOL)

    Δt = 0.1 * min(hx, hy)
    nt = ceil(T / Δt)
    Δt = T / nt
    Wn, ranks, ts = time_loop(W0L, Dx, Dy, X, Y, Δt, T, p, TOL)
    plot(ts, ranks; color=:black)
    display(current())

    heatmap(todense(Wn))
    display(current())
end