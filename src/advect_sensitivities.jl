# Solve the variable-coefficient advection equation
#
#     u_t + a(x) u_x = 0,    x ∈ [xmin, xmax),   t > 0
#     u(x, 0) = u_0(x)
#
# with periodic boundary conditions, using a periodic SBP
# finite-difference derivative from SummationByPartsOperators.jl
# for the spatial discretization and the classical explicit
# Runge–Kutta method of order 4 (RK4) with equidistant time
# steps for the time integration.
#
# Run from the package root:
#   julia --project=. examples/lab2/advect_sensitivities.jl
#
# Required packages (add to the project if not already present):
#   ] add SummationByPartsOperators LinearAlgebra Plots

using LinearAlgebra
using LowRankMethods
using SummationByPartsOperators

# Set to `true` to display the solution after every RK4 step.
# Requires Plots.jl to be available in the active environment.
const DO_PLOT = true
const PLOT_STRIDE = 10            # plot every PLOT_STRIDE-th step

# Set to `true` to use the periodic upwind SBP operator (`Du.minus`,
# left-biased — correct for waves moving right, i.e. a(x) > 0) instead
# of the central periodic SBP derivative.
const USE_UPWIND = true
if DO_PLOT
    using Plots
end

# --- problem setup ---------------------------------------------------------

const xmin = -5.0
const xmax = 5.0
const N = 500                  # number of grid points
const L = xmax - xmin           # period of the domain
const xc = (xmin + xmax) / 2     # center of the domain

# Variable wave speed a(x) (must be periodic on [xmin, xmax])
a_fun(x) = 1.0 - 0.3 * exp(-36 * ((x - 2.5) / (2.5))^2)

# Initial data u_0(x) (must be periodic on [xmin, xmax])
u0_fun(x, a) = exp(-a * (x - xc)^2) +
               exp(-a * (x - xc - L)^2) +
               exp(-a * (x - xc + L)^2)

# --- spatial discretization ------------------------------------------------

# Periodic SBP first-derivative operator. When `USE_UPWIND` is true we take
# the left-biased upwind operator `Du.minus` (appropriate for a(x) > 0);
# otherwise we use the central periodic SBP derivative.
D = if USE_UPWIND
    Du = upwind_operators(periodic_derivative_operator;
        derivative_order=1,
        accuracy_order=4,
        xmin=xmin,
        xmax=xmax,
        N=N)
    Du.minus
else
    periodic_derivative_operator(derivative_order=1,
        accuracy_order=4,
        xmin=xmin,
        xmax=xmax,
        N=N)
end

x = SummationByPartsOperators.grid(D)
a = a_fun.(x)
u0 = u0_fun.(x, 10)

# --- Sensitivity equation -------------------------------------------------
#
# Treat the values `a[j] = a(x_j)` at the grid points as independent
# parameters and let S[:, j] = ∂u/∂a[j].  Differentiating the semi-discrete
# ODE  du/dt = -a .* (D u)  with respect to a[j] gives
#
#     d/dt S[:, j] = -e_j * (D u)[j] - a .* (D S[:, j]),
#
# i.e., column-wise,
#
#     dS/dt = -Diagonal(D u) - a .* (D S).
#
# Initial condition: u_0 is independent of a, so S(0) = 0.
#
# Right-hand side of the joint semi-discrete system. `ux` holds D u and is
# reused for the diagonal source term in the sensitivity equation; `DS` is
# scratch space for D S.
function rhs!(du, dS, u, S, D, a, ux, DS)
    mul!(ux, D, u)              # ux ← D u
    @. du = -a * ux             # du ← -a(x) .* u_x
    @inbounds for j in axes(S, 2)        # DS ← D S, column by column
        mul!(view(DS, :, j), D, view(S, :, j))
    end
    @. dS = -a * DS             # dS ← -a(x) .* (D S), broadcast over columns
    @inbounds for i in eachindex(ux)
        dS[i, i] -= ux[i]       # add -Diagonal(D u)
    end
    return nothing
end

function low_rank_rhs!(u, T::LLRSVD, D, a, dt, r)
    du = -a .* D * u
    left_action = -diag(a) .* D * T.U
    diag_forcing = -diag(D * u)
    FU = qr(left_action, ColumnNorm())
    FV = qr([A.V B.V], ColumnNorm())
    Uhat, Shat, Vhat = svd((FU.R * FU.P') * diagm([A.S; B.S]) * (FV.P * FV.R'))
    return truncate(LLRSVD(FU.Q * Uhat, Shat, FV.Q * Vhat), TOL)
    trunc_sum()
    dX = LLRSVD()
    return du, dX
end

# --- time integration: classical RK4 with equidistant steps ----------------

t0 = 0.0
tf = 10.0

# Pick dt from the advective CFL condition max(a) * dt / h ≈ 1, then round
# Nt up so the equidistant steps land exactly on tf.
h = step(x)                         # uniform grid spacing
amax = maximum(abs, a)
dt_cfl = h / amax
Nt = ceil(Int, (tf - t0) / dt_cfl)
dt = (tf - t0) / Nt
cfl = amax * dt / h

u = copy(u0)
S = zeros(eltype(u0), length(u0), length(u0))   # column j is ∂u/∂a[j]

# Cost functional J(u(t), u_0) = 0.5 * || u(t) - u_0 ||_2^2 (discrete l2 norm).
J(uvec, u0vec) = 0.5 * sum(abs2, uvec .- u0vec)

ux = similar(u)
k1u = similar(u);
k2u = similar(u);
k3u = similar(u);
k4u = similar(u);
utmp = similar(u)

DS = similar(S)
k1S = similar(S);
k2S = similar(S);
k3S = similar(S);
k4S = similar(S);
Stmp = similar(S)

function full_rank_rk4(u, D, S, a, dt)
    rhs!(k1u, k1S, u, S, D, a, ux, DS)
    @. utmp = u + 0.5 * dt * k1u
    @. Stmp = S + 0.5 * dt * k1S
    rhs!(k2u, k2S, utmp, Stmp, D, a, ux, DS)
    @. utmp = u + 0.5 * dt * k2u
    @. Stmp = S + 0.5 * dt * k2S
    rhs!(k3u, k3S, utmp, Stmp, D, a, ux, DS)
    @. utmp = u + dt * k3u
    @. Stmp = S + dt * k3S
    rhs!(k4u, k4S, utmp, Stmp, D, a, ux, DS)
    @. u = u + (dt / 6) * (k1u + 2 * k2u + 2 * k3u + k4u)
    @. S = S + (dt / 6) * (k1S + 2 * k2S + 2 * k3S + k4S)
end

function step_and_truncate_rk4(u, U, S, V, D, a, dt, r)

    u_1 = u + 0.5 * dt
end

# Column of S to display (sensitivity w.r.t. a at this grid index).
const J_PLOT = cld(length(u0), 4)

umin, umax = extrema(u0)
pad = 0.1 * (umax - umin + eps())

ϵ = [1e-4, 1e-6, 1e-8, 1e-10]
cutoff_sv = zeros(length(ϵ))
rϵ = []
t_plot = []

for n in 1:Nt
    full_rank_rk4(u, D, S, a, dt)

    if DO_PLOT && (n % PLOT_STRIDE == 0 || n == Nt)
        sj = @view S[:, J_PLOT]

        # (1,1) solution u(x, t)
        plt_u = plot(x, u;
            label="u(x, t)",
            lw=2,
            xlabel="x",
            ylabel="u",
            title="t = $(round(n*dt; digits=4))",
            ylim=(umin - pad, umax + pad),
            legend=:topright)

        # (1,2) singular value spectrum of S (normalized by σ₁)
        sv = svdvals(S)
        # Clamp singular values to the plot floor so log-scale never sees 0.
        sv_plot = max.(sv, 1e-16)
        plt_sv = plot(1:length(sv_plot), sv_plot ./ sv_plot[1];
            label="σ_k(S) / σ_1",
            lw=2,
            marker=:circle,
            ms=3,
            xlabel="k",
            ylabel="singular value",
            yscale=:log10,
            ylim=(1e-16, 1.0),
            xlim=(1, length(sv_plot)),
            legend=:topright,
            color=:green)

        @info sum(sv_plot ./ sv_plot[1] .≥ 1e-3)
        sv = svdvals(S)
        # Clamp singular values to the plot floor so log-scale never sees 0.
        sv_plot = max.(sv, 1e-16)
        for i in 1:length(ϵ)
            index = sum(sv_plot ./ sv_plot[1] .> ϵ[i])
            cutoff_sv[i] = index
        end
        push!(rϵ, deepcopy(cutoff_sv))
        push!(t_plot, round(n * dt; digits=4))

        # (2,1) sensitivity at one column
        plt_sj = plot(x, sj;
            label="∂u/∂a[$(J_PLOT)] (x_j = $(round(x[J_PLOT]; digits=3)))",
            lw=2,
            xlabel="x",
            ylabel="sensitivity",
            legend=:topright,
            ylim=(-0.1, 0.1),
            color=:red)

        # (2,2) heatmap of the sensitivity matrix S
        smax = maximum(abs, S)
        smax = smax > 0 ? smax : 1.0
        plt_S = heatmap(x, x, S;
            xlabel="x_j  (parameter index)",
            ylabel="x_i  (response point)",
            title="S = ∂u/∂a",
            c=:balance,
            clims=(-smax, smax),
            yflip=true)

        display(plot(plt_u, plt_sv, plt_sj, plt_S;
            layout=(2, 2), size=(1300, 900)))
    end
end

plot(; xlabel="Time", ylabel="r_ϵ", title="Numerical rank", legend=:right)
for i in 1:length(ϵ)
    plot!(t_plot, [r[i] for r in rϵ], label="ϵ = $(ϵ[i])")
end
savefig("lab2_numerical_rank.png")

# --- post-processing -------------------------------------------------------

println("Solved variable-coefficient advection on N = $N points (h = $h).")
println("RK4 with Nt = $Nt steps, dt = $dt, T = $tf, CFL = max(a)·dt/h = $cfl.")
println("‖u(T)‖₂              = ", norm(u))
println("min/max u(T)         = ", extrema(u))
println("‖S(T)‖_F             = ", norm(S))
println("‖∂u/∂a[$(J_PLOT)](T)‖₂     = ", norm(@view S[:, J_PLOT]))

# Gradient of J(u(T), u_0) = 0.5 * ||u(T) - u_0||^2 with respect to the
# parameter vector a, via the chain rule and the sensitivity matrix S:
#     ∂J/∂a[j] = (u(T) - u_0)^T * ∂u(T)/∂a[j] = (u(T) - u_0)^T * S[:, j],
# i.e. ∇_a J = S' * (u(T) - u_0).
gradJ = S' * (u .- u0)
println("‖∇_a J(T)‖_2         = ", norm(gradJ))
println("max|∇_a J(T)|        = ", maximum(abs, gradJ))
println("argmax|∇_a J(T)|     = j = $(argmax(abs.(gradJ)))  (x_j = $(x[argmax(abs.(gradJ))]))")


if DO_PLOT
    display(plot(x, gradJ;
        label="∇_a J(T)",
        lw=2,
        xlabel="x_j",
        ylabel="∂J/∂a[j]",
        title="Gradient of J via S' * (u(T) - u_0)",
        legend=:topright,
        color=:purple,
        size=(900, 400)))
end

