

function applyL(W::Matrix{Float64}, Dx, Dy, X, Y)::Matrix{Float64}
    return Dx * W * Y - X * W * Dy'
end

function applyL(W::LLRSVD, Dx, Dy, X, Y, TOL::Float64)::LLRSVD
    return trunc_sum(
        LLRSVD(Dx * W.U, W.S, Y * W.V),
        LLRSVD(-X * W.U, W.S, Dy * W.V),
        TOL,
    )
end

function W(k::Int, W0::Matrix{T}, Dx, Dy, X, Y)::Vector{Matrix{T}} where T
    Ws = [W0]::Vector{Matrix{T}}
    for p in 1:k
        push!(Ws, applyL(Ws[end], Dx, Dy, X, Y))
    end
    return Ws
end

function W(p::Int, W0::LLRSVD{T}, Dx, Dy, X, Y, TOL::Float64)::Vector{LLRSVD{T}} where T
    Ws = [W0]::Vector{LLRSVD{T}}
    for k in 1:p
        push!(Ws, applyL(Ws[end], Dx, Dy, X, Y, TOL))
    end
    return Ws
end

function taylor_step(
    Wold::LLRSVD{T},
    Dx,
    Dy,
    X,
    Y,
    Δt::Float64,
    p::Int,
    TOL::Float64
)::Vector{LLRSVD{T}} where T
    Ws = W(p, Wold, Dx, Dy, X, Y, 0.0)

    for k in 1:p
        Ws[k+1].S = (Δt)^k / factorial(k) * Ws[k+1].S
        Ws[k+1] = truncate(Ws[k+1], (Δt)^(-k) * TOL)
    end
    return Ws
end

function taylor_step(
    Wold::Matrix{T},
    Dx,
    Dy,
    X,
    Y,
    Δt::Float64,
    p::Int,
)::Matrix{T} where T
    Ws = W(p, Wold, Dx, Dy, X, Y)

    for k in 1:p
        Wold += (Δt)^k / factorial(k) * Ws[k+1]
    end
    return Wold
end

function time_loop(W0::LLRSVD, Dx, Dy, X, Y, Δt::Float64, T::Float64, p::Int, TOL::Float64)
    Wn = W0
    t = 0.0
    ranks = [W0.r]
    ts = [t]
    while t < T
        Ts = taylor_step(Wn, Dx, Dy, X, Y, Δt, p, TOL)
        Wnew = trunc_sum(Ts, Δt^(p + 1))
        @info Wnew.r
        t += Δt
        Wn = Wnew
        push!(ts, t)
        push!(ranks, Wn.r)
    end
    return Wn, ranks, ts
end

function time_loop(W0::Matrix, Dx, Dy, X, Y, Δt::Float64, T::Float64, p::Int)
    Wn = W0
    t = 0.0
    ts = [t]
    while t < T
        Wnew = taylor_step(Wn, Dx, Dy, X, Y, Δt, p)
        t += Δt
        Wn = Wnew
        push!(ts, t)
    end
    return Wn, ts
end