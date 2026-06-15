module LowRankMethods

using LinearAlgebra
using Plots
using LaTeXStrings

function stencil_matrix(h, N)::Matrix{Float64}
    return 1 / (2 * h) .* diagm(
        1 => ones(N - 1),
        -1 => -1 * ones(N - 1),
        N - 1 => [-1.0],
        1 - N => [1.0],
    )
end

include("LLRSVD.jl")
include("tensors.jl")
include("lab1.jl")
include("lab2.jl")
include("lab3.jl")

end # module LowRankMethods
