using Plots
using LinearAlgebra
using LowRankMethods
using LLR
using Pseudospectra

export time_matrix, space_matrix
export initial_condition, spike
export eigenvalue_plot, show_low_rank
export crossDEIM
export solve_sylvester
export check_single_rank
export approximate
export eigensolver, eigensolver_lr
export plot_pseudospectra

function ψ(t, x; α=0.0)
    return 1 / (2 * sqrt(π * t)) * exp(-(x - α)^2 / (4 * t))
end

function multi_green(t, x)
    return ψ(t, x) + ψ(t, x; α=0.3) + ψ(t, x; α=-0.3)
end

function show_low_rank()
    x = LinRange(-1, 1, 100)
    t = LinRange(1e-3, 5, 500)

    u = [ψ(τ, χ) for τ = t, χ = x]
    u = [multi_green(τ, χ) for τ = t, χ = x]

    LU = LLRSVD(u, 0.0)

    p1 = surface(t, x, (t, x) -> multi_green(t, x))
    p2 = plot(LU.S; yscale=:log10, m=:dot, color=:black)
    display(plot(p1, p2; size=(1000, 400)))
    @info (length(LLRSVD(u, 1e-13).S))
end

function time_matrix(nt, Δt)::Matrix{Float64}
    upper_diag = 0.5 * ones(nt - 1)
    main_diag = zeros(nt)
    main_diag[end] = 1.0
    lower_diag = (-0.5) * ones(nt - 1)
    lower_diag[end] = -1.0
    return 1 / Δt .* diagm(
        1 => upper_diag,
        0 => main_diag,
        -1 => lower_diag,
    )
end

function space_matrix(nx, Δx)::Matrix{Float64}
    off_diag = ones(nx - 1)
    main_diag = (-2) * ones(nx)
    return 1 / (Δx^2) .* diagm(
        1 => off_diag,
        0 => main_diag,
        -1 => off_diag,
    )
end

function spike(x; α=0.0)
    if x ≈ α
        return 10.0
    end
    return 0.0
end

function gaussian(x; μ=0.0, σ=1.0)
    return 1 / sqrt(2π * σ^2) * exp(-(x - μ)^2 / (σ^2))
end

function initial_condition(g, nx, nt, L, T)
    Δt = T / nt
    @assert abs(g(0)) < 1e-14
    @assert abs(g(L)) < 1e-14
    x = LinRange(L / (nx + 1), L - L / (nx + 1), nx)
    gx = g.(x)
    e1 = zeros(nt)
    e1[1] = 1.0
    # @info maximum(gx)
    return 1 / (2 * Δt) * (e1 * gx')
end

function rhs_vectors(g, nx, nt, L, T)
    Δt = T / nt
    @assert abs(g(0)) < 1e-14
    @assert abs(g(L)) < 1e-14
    x = LinRange(L / (nx + 1), L - L / (nx + 1), nx)
    gx = g.(x)
    e1 = zeros(nt)
    e1[1] = 1.0
    # @info maximum(gx)
    return 1 / (2 * Δt) * e1, gx
end

function eigenvalue_plot(nt, nx)
    T = 1
    L = 1
    B = time_matrix(nt, T / nt)
    A = space_matrix(nx, L / nx)
    λA = Vector{ComplexF64}(eigvals(A))
    λB = Vector{ComplexF64}(eigvals(B))
    p1 = scatter(λA; title="λ(A)", color=:black, legend=false, ylim=[-1, 1])
    p2 = scatter(λB; title="λ(B)", color=:black, legend=false)
    display(plot(p1, p2; size=(800, 400)))
    display(current())
end

function solve_sylvester(nx, nt; g=nothing)
    L = 1
    T = 1
    if isnothing(g)
        g = x -> gaussian(x; μ=0.5, σ=0.01)
    end
    A = space_matrix(nx, L / (nx + 1))
    B = time_matrix(nt, T / nt)
    C = initial_condition(g, nx, nt, L, T)
    W = sylvester(B, -A', -C)
    LU = LowRankMethods.LLRSVD(W, 0.0)
    plot(LU.S; yscale=:log10, m=:dot, color=:black)
    display(current())
    @info (length(LowRankMethods.LLRSVD(W, 1e-13).S))
    return W
end


function approximate(W::AbstractMatrix)
    U0, _, V0 = svd(W)
    U0 = U0[:, 1:1]
    S0 = [1.0]
    V0 = V0[:, 1:1]
    gfun = (i, j) -> W[i, j]
    U, S, V, info = crossDEIM(gfun, U0, S0, V0)
    @info info
    @info norm(U * diagm(S) * V' - W)
    return U, S, V
end

function check_single_rank()
    g0 = x -> sin(2π * x)
    utrue = (t, x) -> g0(x) * exp(-(2π)^2 * t)
    nx = 50
    nt = 50
    W = solve_sylvester(nx, nt; g=g0)
    U = [utrue(t, x) for t = LinRange(1 / nt, 1, nt), x = LinRange(1 / nx, 1 - 1 / nx, nx)]
    surface(abs.(W - U))
    surface(W)
    display(current())
    return nothing
end

function eigensolver(nx, nt; g=nothing)
    L = 1
    T = 1
    if isnothing(g)
        g = x -> gaussian(x; μ=0.5, σ=0.01)
    end

    A = space_matrix(nx, L / (nx + 1))
    ΛA, Z = eigen(A)
    @assert Z * Z' ≈ 1.0I
    @assert Z * diagm(ΛA) * Z' ≈ A

    B = time_matrix(nt, T / nt)
    ΛB, R = eigen(B)

    C = initial_condition(g, nx, nt, L, T)
    C_hat = R \ (C * Z)

    W_hat = [C_hat[i, j] / (ΛB[i] - ΛA[j]) for i = 1:nt, j = 1:nx]
    W = R * W_hat * Z'
    @assert real.(W) ≈ W

    W = real.(W)
    LU = LowRankMethods.LLRSVD(W, 0.0)
    plot(LU.S; yscale=:log10, m=:dot, color=:black)
    display(current())
    @info (length(LowRankMethods.LLRSVD(W, 1e-13).S))
    return W
end

function eigensolver_lr(nx, nt; g=nothing)
    L = 1
    T = 1
    if isnothing(g)
        g = x -> gaussian(x; μ=0.5, σ=0.01)
    end

    A = space_matrix(nx, L / (nx + 1))
    ΛA, Z = eigen(A)
    @assert Z * Z' ≈ 1.0I
    @assert Z * diagm(ΛA) * Z' ≈ A

    B = time_matrix(nt, T / nt)
    ΛB, R = eigen(B)

    C, D = rhs_vectors(g, nx, nt, L, T)
    C = reshape(C, (:, 1))
    D = reshape(D, (:, 1))
    C_hat = R \ C
    D_hat = (D' * Z)'

    Wparam = (i, j) -> (C_hat[i] * D_hat[j]) / (ΛB[i] - ΛA[j])
    opts = (tol=1e-20, r_max=15, r_in=15, max_iter=30)
    U, S, V, info = crossDEIM(Wparam, C_hat, [1.0], D_hat, opts)
    @info info
    W_lowrank = R * U * diagm(S) * (Z * V)'
    @assert real.(W_lowrank) ≈ W_lowrank
    W_lowrank = real.(W_lowrank)

    LU_lowrank = LowRankMethods.LLRSVD(W_lowrank, 0.0)
    plot(LU_lowrank.S; yscale=:log10, m=:dot, color=:black)
    display(current())
    @info (length(LowRankMethods.LLRSVD(W_lowrank, 1e-13).S))
    return W_lowrank
end

function plot_pseudospectra()
    B = time_matrix(10, 0.1)
    Blarge = time_matrix(100, 0.01)
    p1 = spectralportrait(B; label="nt=10")
    p2 = spectralportrait(Blarge; label="nt=100")
    display(plot(p1, p2; size=(1000, 400)))
end
