export applyL
export W
export taylor_step
export time_loop

function applyL(W::Tucker3, Dx::AbstractMatrix, Dy::AbstractMatrix, Dz::AbstractMatrix, tol::Float64)::Tucker3
    LWx = Tucker3(W.G, Dx * W.U1, W.U2, W.U3)
    LWy = Tucker3(W.G, W.U1, Dy * W.U2, W.U3)
    LWz = Tucker3(W.G, W.U1, W.U2, Dz * W.U3)
    return tucker_sum([LWx, LWy, LWz], tol)
end

function W(k::Int, W0::Tucker3, Dx, Dy, Dz, TOL::Float64)::Tucker3
    if k == 0
        return W0
    end
    return applyL(W(k - 1, W0, Dx, Dy, Dz, TOL), Dx, Dy, Dz, TOL)
end


function taylor_step(
    Wold::Tucker3,
    Dx::AbstractMatrix,
    Dy::AbstractMatrix,
    Dz::AbstractMatrix,
    Δt::Float64,
    p::Int,
    TOL::Float64;
)::Vector{Tucker3}
    Ts = [Wold]::Vector{Tucker3}
    for k in 1:p
        Wk = W(k, Wold, Dx, Dy, Dz, TOL * Δt^(-k))
        push!(Ts, Tucker3((Δt)^k / factorial(k) * Wk.G, Wk.U1, Wk.U2, Wk.U3))
    end
    return Ts
end


function time_loop(
    W0::Tucker3,
    Dx::AbstractMatrix,
    Dy::AbstractMatrix,
    Dz::AbstractMatrix,
    Δt::Float64, T::Float64,
    p::Int,
    TOL::Float64;
)
    Wn = W0
    t = 0.0
    rs = []::Vector{}
    while t < T
        Wnew = tucker_sum(taylor_step(Wn, Dx, Dy, Dz, Δt, p, TOL), Δt^(p + 1))
        push!(rs, size(Wnew.G))
        t += Δt
        Wn = Wnew
    end
    return Wn, rs
end
