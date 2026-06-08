export LLRSVD, todense, toelem, truncate, trunc_sum

mutable struct LLRSVD{T<:Number}
    U::Matrix{T}
    S::Vector{T}
    V::Matrix{T}
    m::Int
    n::Int
    r::Int
end

function LLRSVD(U::AbstractMatrix, S::AbstractVector, V::AbstractMatrix)
    T = promote_type(eltype(U), eltype(S), eltype(V))
    LLRSVD{T}(Matrix{T}(U), Vector{T}(S), Matrix{T}(V))
end

function LLRSVD(U::Matrix{T}, S::Vector{T}, V::Matrix{T}) where T
    m, r = size(U)
    n, r = size(V)
    return LLRSVD{T}(U, S, V, m, n, r)
end

function LLRSVD(W::Matrix{T}, TOL::Float64) where T
    U, S, V = svd(W)
    subset = findall(x -> x > TOL, S)
    if isempty(subset)
        return LLRSVD(U[:, [1]], [0.0], V[:, [1]])
    end
    # @assert !isempty(subset)
    return LLRSVD(U[:, subset], S[subset], V[:, subset])
end

function todense(A::LLRSVD{T})::Matrix{T} where T
    return A.U * diagm(A.S) * A.V'
end

function toelem(A::LLRSVD{T}, i::Int, j::Int) where T
    return dot(A.U[i, :] .* A.S, A.V[j, :])
end

function truncate(A::LLRSVD{T}, TOL::Float64)::LLRSVD{T} where T
    ranks_to_keep = A.r - sum(cumsum(reverse(A.S .^ 2)) .≤ TOL^2)
    ranks_to_keep = max(ranks_to_keep, 1)
    return LLRSVD(A.U[:, 1:ranks_to_keep], A.S[1:ranks_to_keep], A.V[:, 1:ranks_to_keep])
end

function trunc_sum(A::LLRSVD{T}, B::LLRSVD, TOL::Float64)::LLRSVD{T} where T
    FU = qr([A.U B.U], ColumnNorm())
    FV = qr([A.V B.V], ColumnNorm())
    Uhat, Shat, Vhat = svd((FU.R * FU.P') * diagm([A.S; B.S]) * (FV.P * FV.R'))
    return truncate(LLRSVD(FU.Q * Uhat, Shat, FV.Q * Vhat), TOL)
end

function trunc_sum(terms::Vector{LLRSVD{T}}, TOL::Float64)::LLRSVD{T} where T
    if length(terms) == 2
        return trunc_sum(terms[1], terms[2], TOL)
    end
    return trunc_sum(terms[1], trunc_sum(terms[2:end], TOL), TOL)
end