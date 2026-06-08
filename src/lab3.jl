
export l3ex1, l3ex2, l3ex3, l3ex4, l3ex5, l3ex6, l3ex7

include("tucker_transport.jl")

function l3ex1()
    A = randn(4, 5, 6)
    @assert size(unfold(A, 1)) == (4, 30)
    @assert fold(unfold(A, 1), size(A), 1) == A
    @assert fold(unfold(A, 2), size(A), 2) == A
    @assert fold(unfold(A, 3), size(A), 3) == A

    M1 = randn(3, 4)
    M2 = randn(5, 5)
    @assert isapprox(mode_product(mode_product(A, M1, 1), M2, 2), mode_product(mode_product(A, M2, 2), M1, 1))

    x = LinRange(0, π, 20)
    A = [x * y * z for x = sin.(x), y = cos.(2 * x), z = exp.(-x)]
    @assert size(A) == (20, 20, 20)
    @assert LLRSVD(unfold(A, 1), 1e-12).r == 1
    @assert LLRSVD(unfold(A, 2), 1e-12).r == 1
    @assert LLRSVD(unfold(A, 3), 1e-12).r == 1
end


function l3ex2()
    x = LinRange(0, π, 20)
    A = [x * y * z for x = sin.(x), y = cos.(2 * x), z = exp.(-x)]
    TA = hosvd(A, (1, 1, 1))
    @assert norm(todense(TA) - A) < 1e-12

    A = randn(30, 30, 30)
    @assert norm(A - todense(Tucker3(A, 1e-12))) < 1e-10 * norm(A)
end

function l3ex3()
    x = vec(1:20)
    A = [1 / (x + y + z) for x = x, y = x, z = x]
    T = Tucker3(A, 0.0)
    @assert size(T.G) == (20, 20, 20)
    @info size(tucker_round(T, 1e-2).G)
    @info size(tucker_round(T, 1e-4).G)
    @info size(tucker_round(T, 1e-8).G)

    x = LinRange(-3, 3, 100)
    A = [exp(-(x^2 + y^2 + z^2)) for x = x, y = x, z = x]
    @info size(Tucker3(A, 1e-12).G)

    x = LinRange(-3, 3, 50)
    A = [exp(-(x^2 + y^2 + z^2)) for x = x, y = x, z = x]
    @info size(Tucker3(A, 1e-12).G)
end

function l3ex4()
    tol = 1e-6
    x = LinRange(0, 2 * π, 50)
    A = [sin(x + y + z) for x = x, y = x, z = x]
    S1 = LLRSVD(unfold(A, 1), 0.0).S
    S2 = LLRSVD(unfold(A, 2), 0.0).S
    S3 = LLRSVD(unfold(A, 3), 0.0).S
    plot(; yscale=:log10, ylabel=L"\sigma_k", xlabel=L"k", title=L"\sin(x+y+z)")
    plot!(S1; m=:xcross, label=L"A_{(1)}", color=:black)
    plot!(S2; m=:circle, label=L"A_{(2)}", color=:black)
    plot!(S3; m=:cross, label=L"A_{(3)}", color=:black)
    display(current())
    savefig("LowRankMethods/reports/figures/lab3_e4_1.png")

    r1 = LLRSVD(unfold(A, 1), tol).r
    r2 = LLRSVD(unfold(A, 2), tol).r
    r3 = LLRSVD(unfold(A, 3), tol).r
    @info "sin function ranks: $r1, $r2, $r3"

    x = LinRange(0, 1, 50)
    ϵ = 0.1
    A = [1 / sqrt(x^2 + y^2 + z^2 + ϵ^2) for x = x, y = x, z = x]
    S1 = LLRSVD(unfold(A, 1), 0.0).S
    S2 = LLRSVD(unfold(A, 2), 0.0).S
    S3 = LLRSVD(unfold(A, 3), 0.0).S
    plot(; yscale=:log10, ylabel=L"\sigma_k", xlabel=L"k", title=L"1/\sqrt{x^2+y^2+z^2+\varepsilon^2}")
    plot!(S1; m=:xcross, label=L"A_{(1)}", color=:black)
    plot!(S2; m=:circle, label=L"A_{(2)}", color=:black)
    plot!(S3; m=:cross, label=L"A_{(3)}", color=:black)
    display(current())
    savefig("LowRankMethods/reports/figures/lab3_e4_2.png")

    r1 = LLRSVD(unfold(A, 1), tol).r
    r2 = LLRSVD(unfold(A, 2), tol).r
    r3 = LLRSVD(unfold(A, 3), tol).r
    @info "sqrt function ranks: $r1, $r2, $r3"

    x = LinRange(-1, 1, 50)
    A = [tanh(10 * (x^2 + y^2 - z)) for x = x, y = x, z = x]
    S1 = LLRSVD(unfold(A, 1), 0.0).S
    S2 = LLRSVD(unfold(A, 2), 0.0).S
    S3 = LLRSVD(unfold(A, 3), 0.0).S
    plot(; yscale=:log10, ylabel=L"\sigma_k", xlabel=L"k", title=L"\tanh\left(10(x^2+y^2-z)\right)")
    plot!(S1; m=:xcross, label=L"A_{(1)}", color=:black)
    plot!(S2; m=:circle, label=L"A_{(2)}", color=:black)
    plot!(S3; m=:cross, label=L"A_{(3)}", color=:black)
    display(current())
    savefig("LowRankMethods/reports/figures/lab3_e4_3.png")

    r1 = LLRSVD(unfold(A, 1), tol).r
    r2 = LLRSVD(unfold(A, 2), tol).r
    r3 = LLRSVD(unfold(A, 3), tol).r
    @info "tanh function ranks: $r1, $r2, $r3"
end

function l3ex5()
    A = randn(3, 4, 5)
    B = randn(3, 4, 5)
    TA = Tucker3(A, 0.0)
    TB = Tucker3(B, 0.0)
    @assert isapprox(sum(A .* B), tucker_inner(TA, TB))
end

function l3ex6()
    x = LinRange(-3, 3, 50)
    A = [exp(-(x^2 + y^2 + z^2)) for x = x, y = x, z = x]
    TA = Tucker3(A, 1e-10)
    @info size(TA.G)

    x = LinRange(0, 2 * π, 50)
    B = [sin(x + y + z) for x = x, y = x, z = x]
    TB = Tucker3(B, 1e-10)
    @info size(TB.G)

    TC = tucker_add(TA, TB)
    @info size(TC.G)

    @info norm((A + B) - todense(TC))

    TAsum = tucker_sum(TA, TA, 1e-10)
    @assert size(TAsum.G) == size(Tucker3(A + A, 1e-10).G)
    @info norm(2 * A - todense(TAsum))

    TAsum = tucker_sum([TA, TA, TA], 1e-10)
    @assert size(TAsum.G) == size(Tucker3(A + A + A, 1e-10).G)
end

function l3ex7()
    m = 32
    Lx = 2 * π
    hx = Lx / m
    Dx = stencil_matrix(hx, m)
    T = 2 * π
    x = LinRange(0, Lx - hx, m)
    TOL = 1e-5
    p = 4
    W0 = [x * y * z for x = sin.(x), y = sin.(x), z = sin.(x)]

    W0L = Tucker3(W0, TOL)

    Δt = 0.001
    nt = Int(ceil(T / Δt))
    Δt = T / nt
    Wexact, _ = time_loop(W0L, Dx, Dx, Dx, Δt, nt, p, TOL)

    Δts = [0.4, 0.2, 0.1, 0.05]
    for Δt in Δts
        nt = Int(ceil(T / Δt))
        Δt = T / nt
        Wn, _ = time_loop(W0L, Dx, Dx, Dx, Δt, nt, p, TOL)
        @info "A: $(hx^(3 / 2) * norm((todense(Wn) - W0)))"
        @info "N: $(hx^(3 / 2) * norm((todense(Wn) - todense(Wexact))))"
    end
    return
end
