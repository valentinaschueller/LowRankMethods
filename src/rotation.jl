

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

function W(k::Int, W0::LLRSVD{T}, Dx, Dy, X, Y, TOL::Float64) where T
    if k == 0
        return W0
    end
    return applyL(W(k - 1, W0, Dx, Dy, X, Y, TOL), Dx, Dy, X, Y, TOL)
end

function taylor_step(
    Wold::LLRSVD{T},
    Dx,
    Dy,
    X,
    Y,
    Δt::Float64,
    p::Int,
    TOL::Float64;
    truncate_before_scaling::Bool=true,
)::Vector{LLRSVD{T}} where T
    Ws = [Wold]::Vector{LLRSVD{T}}
    # @info "0: $(Ts[end].r)"

    for k in 1:p
        Wk_TOL = truncate_before_scaling ? TOL * Δt^(-k) : 0.0
        # Wk = applyL(Ws[k], Dx, Dy, Wk_TOL)
        push!(Ws, W(k, Wold, Dx, Dy, X, Y, Wk_TOL))
    end


    Ts = [Wold]::Vector{LLRSVD{T}}
    for k in 1:p
        push!(Ts, truncate(LLRSVD(Ws[k+1].U, (Δt)^k / factorial(k) * Ws[k+1].S, Ws[k+1].V), TOL))
        # @info "$k: $(Ts[end].S)"
        # @info trunc_sum(Ts, TOL * Δt^(-k)).S
    end
    return Ts
end

function time_loop(W0::LLRSVD, Dx, Dy, X, Y, Δt::Float64, T::Float64, p::Int, TOL::Float64)
    Wn = W0
    t = 0.0
    ranks = [W0.r]
    ts = [t]
    while t < T
        Ts = taylor_step(Wn, Dx, Dy, X, Y, Δt, p, TOL)
        Wnew = trunc_sum(Ts, Δt^(p - 1))
        @info Wnew.r
        t += Δt
        Wn = Wnew
        push!(ts, t)
        push!(ranks, Wn.r)
    end
    return Wn, ranks, ts
end