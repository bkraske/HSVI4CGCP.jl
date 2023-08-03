Base.@kwdef struct SARSOPSolver{LOW,UP,TERM} <: Solver
    epsilon::Float64     = 0.5
    kappa::Float64       = 0.5
    delta::Float64       = 1e-1
    max_time::Float64    = 1.0
    max_steps::Int       = typemax(Int)
    verbose::Bool        = true
    init_lower::LOW      = BlindLowerBound(bel_res = 1e-2)
    init_upper::UP       = FastInformedBound(bel_res=1e-2)
    prunethresh::Float64 = 0.10
    terminate::TERM      = RootDifference(0.001)
    # ρ::Float64          = 3.0
end

struct RootDifference <: Function
    precision::Float64
end

(rd::RootDifference)(tree) = root_diff(tree) <= rd.precision

# below doesn't need to go in this package
struct CGCPRootDifference <: Function
    rho::Float64
    precision::Float64
end

function (c::CGCPRootDifference)(tree)
    ga = 10^(log10(max(abs(tree.V_upper[1]),abs(tree.V_lower[1])))-c.ρ)
    return root_diff(tree) <= ga
end

function solve_info(solver::SARSOPSolver, pomdp::POMDP)
    tree = SARSOPTree(solver, pomdp)

    t0 = time()
    iter = 0
    while time()-t0 < solver.max_time && !solver.terminate(tree)
        sample!(solver, tree)
        backup!(tree)
        prune!(solver, tree)
        iter += 1
        ga = 10^(log10(max(abs(tree.V_upper[1]),abs(tree.V_lower[1])))-solver.ρ)
    end

    pol = AlphaVectorPolicy(
        pomdp,
        getproperty.(tree.Γ, :alpha),
        ordered_actions(pomdp)[getproperty.(tree.Γ, :action)]
    )
    return pol, (;
        time = time()-t0, 
        tree,
        iter
    )
end

POMDPs.solve(solver::SARSOPSolver, pomdp::POMDP) = first(solve_info(solver, pomdp))
