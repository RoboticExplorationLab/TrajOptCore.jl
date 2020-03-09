module TrajOptCore

using Dynamics
using DifferentialRotations
using StaticArrays
using LinearAlgebra
using DocStringExtensions
using ForwardDiff
using UnsafeArrays

import Dynamics: Implicit, Explicit, AbstractKnotPoint, DEFAULT_Q, is_terminal, state_diff, StaticKnotPoint
import Dynamics: jacobian!, error_expansion!, error_expansion, state_dim, control_dim  # extended methods

# problems
export
    Problem,
    initial_controls!,
    initial_states!,
    rollout!

# cost functions
export
    AbstractObjective,
    Objective,
    LQRObjective,
    QuadraticCost,
    LQRCost,
    Expansion,
    CostExpansion,
    cost,
    StaticExpansion,
    cost_expansion!,
    error_expansion!


# constraints
export
    AbstractConstraint,
    ConstraintSet,
    ConstraintParams,
    evaluate!,
    update_active_set!,
    max_violation


include("expansions.jl")
include("costfunctions.jl")
include("objective.jl")

include("abstract_constraint.jl")
include("constraints.jl")
include("dynamics_constraints.jl")
include("integration.jl")

include("cost.jl")
include("constraint_vals.jl")
include("constraint_sets.jl")
include("problem.jl")

include("utils.jl")
end # module
