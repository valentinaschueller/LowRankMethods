module LowRankMethods

using LinearAlgebra

export LLRSVD, todense, toelem, truncate, toeplitz

FT = Float64

# Exercise 1
mutable struct LLRSVD
    U::Matrix{FT}
    S::Vector{FT}
    V::Matrix{FT}
    m::Int
    n::Int
    r::Int
end

function LLRSVD(U::Matrix{FT}, S::Vector{FT}, V::Matrix{FT})
    m, r = size(U)
    n, r = size(V)
    return LLRSVD(U, S, V, m, n, r)
end

function LLRSVD(W::Matrix{FT}, TOL::Float64)
    U, S, V = svd(W)
    subset = findall(x -> x > TOL, S)
    @assert !isempty(subset)
    return LLRSVD(U[:, subset], S[subset], V[:, subset])
end

function todense(A::LLRSVD)::Matrix{FT}
    return A.U * diagm(A.S) * A.V'
end

function toelem(A::LLRSVD, i::Int, j::Int)
    return dot(A.U[i, :] .* A.S, A.V[j, :])
end

W = rand(30, 20)
@assert norm(W - todense(LLRSVD(W, 0.0))) < 1e-12

# Exercise 2

function truncate(A::LLRSVD, TOL::Float64)::LLRSVD
    ranks_to_keep = A.r - sum(cumsum(reverse(A.S .^ 2)) .≤ TOL^2)
    return LLRSVD(A.U[:, 1:ranks_to_keep], A.S[1:ranks_to_keep], A.V[:, 1:ranks_to_keep])
end

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
# @info(A.S)
T = truncate(A, 2e-3)
# @info(T.S)
@info(norm(todense(A) - todense(T)))

# Exercise 3

m = 64
n = 64
Lx = 2 * π
Ly = 2 * π
hx = Lx / m
hy = Ly / n
x = LinRange(0, Lx - hx, m)
y = LinRange(0, Ly - hy, n)
W0 = sin.(x) * sin.(y')

Dx = 1 / hx .* diagm(1 => 0.5 * ones(m - 1), -1 => -0.5 * ones(m - 1), m - 1 => [-0.5], 1 - m => [0.5])
Dy = 1 / hy .* diagm(1 => 0.5 * ones(n - 1), -1 => -0.5 * ones(n - 1), n - 1 => [-0.5], 1 - n => [0.5])

numdiff = Dx * W0 + W0 * Dy'
truediff = cos.(x) * sin.(y)' + sin.(x) * cos.(y)'
@assert (hx * hy * norm(numdiff - truediff) < 1e-2)

end # module LowRankMethods
