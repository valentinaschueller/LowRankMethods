function applyL(W::Matrix{T}, Dx::Matrix{T}, Dy::Matrix{T})::Matrix{T} where T
    return Dx * W + W * Dy'
end


function W(k::Int, W0, Dx, Dy)
    if k == 0
        return W0
    end
    return applyL(W(k - 1, W0, Dx, Dy), Dx, Dy)
end


function taylor_term(k::Int, Δt, W0, Dx, Dy)
    return ((Δt)^k / factorial(k)) * W(k, W0, Dx, Dy)
end


function applyL(W::LLRSVD{T}, Dx, Dy, TOL::Float64)::LLRSVD{T} where T
    return trunc_sum(
        LLRSVD(Dx * W.U, W.S, W.V),
        LLRSVD(W.U, W.S, Dy * W.V),
        TOL,
    )
end

function W(k::Int, W0::LLRSVD{T}, Dx, Dy, TOL::Float64) where T
    if k == 0
        return W0
    end
    return applyL(W(k - 1, W0, Dx, Dy, TOL), Dx, Dy, TOL)
end


function taylor_step(
    Wold::LLRSVD{T},
    Dx,
    Dy,
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
        push!(Ws, W(k, Wold, Dx, Dy, Wk_TOL))
    end


    Ts = [Wold]::Vector{LLRSVD{T}}
    for k in 1:p
        push!(Ts, truncate(LLRSVD(Ws[k+1].U, (Δt)^k / factorial(k) * Ws[k+1].S, Ws[k+1].V), TOL))
        # @info "$k: $(Ts[end].S)"
        # @info trunc_sum(Ts, TOL * Δt^(-k)).S
    end
    return Ts
end


function time_loop_A(W0::LLRSVD, Dx, Dy, Δt::Float64, T::Float64, p::Int, TOL::Float64)
    Wn = W0
    t = 0.0
    while t < T
        Wnew = trunc_sum(taylor_step(Wn, Dx, Dy, Δt, p, TOL), Δt^(p - 1))
        t += Δt
        Wn = Wnew
        @info "t=$t, Rank: $(Wnew.r)"
        # @assert Wnew.r == 1
    end
    return Wn
end

function time_loop_B(W0::LLRSVD, Dx, Dy, Δt::Float64, T::Float64, p::Int, TOL::Float64)
    Wn = W0
    t = 0.0
    while t < T
        Wnew = trunc_sum(taylor_step(Wn, Dx, Dy, Δt, p, TOL; truncate_before_scaling=false), Δt^(p + 1))
        t += Δt
        Wn = Wnew
    end
    return Wn
end
