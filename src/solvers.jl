
"""
    AbstractSolver{T} <: MathOptInterface.AbstractNLPEvaluator

Abstract solver for trajectory optimization problems

# Interface
Any type that inherits from `AbstractSolver` must define the following methods:
```julia
model = get_model(::AbstractSolver)::AbstractModel
obj   = get_objective(::AbstractSolver)::AbstractObjective
E     = get_cost_expansion(::AbstractSolver)::QuadraticExpansion  # quadratic error state expansion
Z     = get_trajectory(::AbstractSolver)::Traj
n,m,N = Base.size(::AbstractSolver)
x0    = get_initial_state(::AbstractSolver)::StaticVector
solve!(::AbstractSolver)
```

Optional methods for line search and merit function interface. Note that these do not
    have to return `Traj`
```julia
Z     = get_solution(::AbstractSolver)  # current solution (defaults to get_trajectory)
Z     = get_primals(::AbstractSolver)   # current primals estimate used in the line search
dZ    = get_step(::AbstractSolver)      # current step in the primal variables
```

Optional methods
```julia
opts  = options(::AbstractSolver)       # options struct for the solver. Defaults to `solver.opts`
st    = solver_stats(::AbstractSolver)  # dictionary of solver statistics. Defaults to `solver.stats`
iters = iterations(::AbstractSolver)    #
```
"""
abstract type AbstractSolver{T} <: MOI.AbstractNLPEvaluator end

# Default getters
@inline get_model(solver::AbstractSolver) = solver.model
@inline get_objective(solver::AbstractSolver) = solver.obj
@inline get_cost_expansion(solver::AbstractSolver) = solver.E
@inline get_trajectory(solver::AbstractSolver) = solver.Z
@inline get_initial_state(solver::AbstractSolver) = solver.x0

@inline get_solution(solver::AbstractSolver) = get_trajectory(solver)
@inline get_primals(solver::AbstractSolver) = solver.Z̄
@inline get_step(solver::AbstractSolver) = solver.δZ
@inline options(solver::AbstractSolver) = solver.opts
@inline stats(solver::AbstractSolver) = solver.stats
iterations(solver::AbstractSolver) = stats(solver).iterations


"$(TYPEDEF) Unconstrained optimization solver. Will ignore
any constraints in the problem"
abstract type UnconstrainedSolver{T} <: AbstractSolver{T} end


"""$(TYPEDEF)
Abstract solver for constrained trajectory optimization problems

In addition to the methods required for `AbstractSolver`, all `ConstrainedSolver`s
    must define the following method
```julia
get_constraints(::ConstrainedSolver)::ConstrainSet
```
"""
abstract type ConstrainedSolver{T} <: AbstractSolver{T} end




function cost(solver::AbstractSolver, Z=get_trajectory(solver))
    obj = get_objective(solver)
    cost(obj, Z)
end

function RobotDynamics.rollout!(solver::AbstractSolver)
    Z = get_trajectory(solver)
    model = get_model(solver)
    x0 = get_initial_state(solver)
    rollout!(model, Z, x0)
end

RobotDynamics.states(solver::AbstractSolver) = [state(z) for z in get_trajectory(solver)]
function RobotDynamics.controls(solver::AbstractSolver)
    N = size(solver)[3]
    Z = get_trajectory(solver)
    [control(Z[k]) for k = 1:N-1]
end

set_initial_state!(solver, x0) = copyto!(get_initial_state(solver), x0)

@inline TrajOptCore.initial_states!(solver::AbstractSolver, X0) = set_states!(get_trajectory(solver), X0)
@inline TrajOptCore.initial_controls!(solver::AbstractSolver, U0) = set_controls!(get_trajectory(solver), U0)
function TrajOptCore.initial_trajectory!(solver::AbstractSolver, Z0::Traj)
    Z = get_trajectory(solver)
    for k in eachindex(Z)
        Z[k].z = copy(Z0[k].z)
    end
end

# Default getters
@inline RobotDynamics.get_times(solver::AbstractSolver) = RobotDynamics.get_times(get_trajectory(solver))


# Line Search and Merit Functions
"""
    get_primals(solver)
    get_primals(solver, α::Real)

Get the primal variables used during the line search. When called without an `α` it should
return the container where the temporary primal variables are stored. When called with a
step length `α` it returns `z + α⋅dz` where `z = get_solution(solver)` and `dz = get_step(solver)`.
"""
function get_primals(solver::AbstractSolver, α)
    z = get_solution(solver)
    z̄ = get_primals(solver)
    dz = get_step(solver)
    z̄ .= z .+ α*dz
end

# Constrained solver
TrajOptCore.num_constraints(solver::AbstractSolver) = num_constraints(get_constraints(solver))

function TrajOptCore.max_violation(solver::ConstrainedSolver, Z::Traj=get_trajectory(solver))
    conSet = get_constraints(solver)
    evaluate!(conSeti, Z)
    max_violation(solver)
end

@inline TrajOptCore.findmax_violation(solver::ConstrainedSolver) =
    findmax_violation(get_constraints(solver))
