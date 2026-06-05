export unfold, fold, mode_product, hosvd
export tucker_add
export tucker_round
export tucker_sum
export tucker_norm
export tucker_inner
export Tucker3

function unfold(A::Array{Float64,3}, n::Int)::Matrix{Float64}
    n1, n2, n3 = size(A)
    if n == 1
        return reshape(permutedims(A, [1, 2, 3]), n1, n2 * n3)
    end
    if n == 2
        return reshape(permutedims(A, [2, 1, 3]), n2, n1 * n3)
    end
    if n == 3
        return reshape(permutedims(A, [3, 1, 2]), n3, n1 * n2)
    end
    throw("n must be in {1,2,3}!")
end

function fold(An::Matrix{Float64}, tensor_size::Tuple, n::Int)::Array{Float64,3}
    n1, n2, n3 = tensor_size
    if n == 1
        return permutedims(reshape(An, (n1, n2, n3)), [1, 2, 3])
    end
    if n == 2
        return permutedims(reshape(An, (n2, n1, n3)), [2, 1, 3])
    end
    if n == 3
        return permutedims(reshape(An, (n3, n1, n2)), [2, 3, 1])
    end
    throw("n must be in {1,2,3}!")
end

function mode_product(A::Array{Float64,3}, M::AbstractMatrix, n::Int)::Array{Float64,3}
    newsize = setindex!(collect(size(A)), size(M)[1], n)
    return fold(M * unfold(A, n), Tuple(newsize), n)
end

mutable struct Tucker3
    G::Array{Float64,3}
    U1::AbstractMatrix
    U2::AbstractMatrix
    U3::AbstractMatrix
end

function todense(T::Tucker3)::Array{Float64,3}
    return mode_product(mode_product(mode_product(T.G, T.U1, 1), T.U2, 2), T.U3, 3)
end

function hosvd(A::Array{Float64,3}, r::Tuple{Int,Int,Int})::Tucker3
    U1 = LLRSVD(unfold(A, 1), 0.0).U[:, 1:r[1]]
    U2 = LLRSVD(unfold(A, 2), 0.0).U[:, 1:r[2]]
    U3 = LLRSVD(unfold(A, 3), 0.0).U[:, 1:r[3]]
    G = mode_product(mode_product(mode_product(A, U1', 1), U2', 2), U3', 3)
    return Tucker3(G, U1, U2, U3)
end

function Tucker3(A::Array{Float64,3}, tol::Float64)::Tucker3
    U1 = LLRSVD(unfold(A, 1), tol / sqrt(3)).U
    U2 = LLRSVD(unfold(A, 2), tol / sqrt(3)).U
    U3 = LLRSVD(unfold(A, 3), tol / sqrt(3)).U
    G = mode_product(mode_product(mode_product(A, U1', 1), U2', 2), U3', 3)
    return Tucker3(G, U1, U2, U3)
end

function tucker_round(T::Tucker3, tol::Float64)::Tucker3
    Q, R = qr(T.U1)
    T.U1 = Matrix(Q)
    T.G = mode_product(T.G, R, 1)
    Q, R = qr(T.U2)
    T.U2 = Matrix(Q)
    T.G = mode_product(T.G, R, 2)
    Q, R = qr(T.U3)
    T.U3 = Matrix(Q)
    T.G = mode_product(T.G, R, 3)
    GT = Tucker3(T.G, tol)
    return Tucker3(GT.G, T.U1 * GT.U1, T.U2 * GT.U2, T.U3 * GT.U3)
end

function tucker_inner(A::Tucker3, B::Tucker3)::Float64
    U1 = A.U1' * B.U1
    U2 = A.U2' * B.U2
    U3 = A.U3' * B.U3
    H = mode_product(mode_product(mode_product(B.G, U1, 1), U2, 2), U3, 3)
    @assert size(H) == size(A.G)
    return sum(A.G .* H)
end

function tucker_norm(A::Tucker3)::Float64
    return tucker_inner(A, A)
end

function tucker_add(A::Tucker3, B::Tucker3)::Tucker3
    q1, r1, s1 = size(A.G)
    q2, r2, s2 = size(B.G)
    G = zeros((q1 + q2, r1 + r2, s1 + s2))
    G[1:q1, 1:r1, 1:s1] = A.G
    G[q1+1:end, r1+1:end, s1+1:end] = B.G
    Tucker3(G, [A.U1 B.U1], [A.U2 B.U2], [A.U3 B.U3])
end

function tucker_sum(A::Tucker3, B::Tucker3, tol::Float64)::Tucker3
    return tucker_round(tucker_add(A, B), tol)
end

function tucker_sum(terms::Vector{Tucker3}, tol::Float64)::Tucker3
    if length(terms) == 2
        return tucker_sum(terms[1], terms[2], tol)
    end
    return tucker_sum(terms[1], tucker_sum(terms[2:end], tol), tol)
end