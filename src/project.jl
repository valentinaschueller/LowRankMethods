using Plots
using LinearAlgebra
using LowRankMethods

function ψ(t, x; α=0.0)
    return 1 / (2 * sqrt(π * t)) * exp(-(x - α)^2 / (4 * t))
end

function multi_green(t, x)
    return ψ(t, x) + ψ(t, x; α=0.3) + ψ(t, x; α=-0.3)
end

x = LinRange(-1, 1, 100)
t = LinRange(1e-3, 5, 500)

T = [τ for τ = t, χ = x]
X = [χ for τ = t, χ = x]
u = [ψ(τ, χ) for τ = t, χ = x]
u = [multi_green(τ, χ) for τ = t, χ = x]

LU = LLRSVD(u, 0.0)

p1 = surface(t, x, (t, x) -> multi_green(t, x))
p2 = plot(LU.S; yscale=:log10, m=:dot, color=:black)
display(plot(p1, p2; size=(1000, 400)))
@info (length(LLRSVD(u, 1e-13).S))

