module TrajOptCore

using RobotDynamics
# using Rotations
using StaticArrays
using LinearAlgebra
using DocStringExtensions
using ForwardDiff
using UnsafeArrays

import RobotDynamics: Implicit, Explicit, AbstractKnotPoint, DEFAULT_Q, is_terminal, state_diff,
	StaticKnotPoint, state_diff_jacobian!, state_diff_jacobian
import RobotDynamics: jacobian!, state_dim, control_dim  # extended methods

# re-export
import RobotDynamics: KnotPoint
export
	KnotPoint,
	Traj,
	rollout!

# problems
export
    Problem,
    initial_controls!,
    initial_states!,
	initial_trajectory!,
	set_initial_state!,
    rollout!,
	integration,
	change_integration,
	get_constraints,
	get_model,
	get_objective,
	get_trajectory

# cost functions
export
    AbstractObjective,
    Objective,
    LQRObjective,
	QuadraticObjective,
	CostFunction,
    QuadraticCost,
	DiagonalCost,
    LQRCost,
    LQRCostTerminal,
    cost

# cost expansion
export
    CostExpansion,
    Expansion,
    StaticExpansion,
    cost_expansion!,
    cost_hessian!,
    cost_gradient!,
    error_expansion!,
    error_expansion

# constraint types
export
    AbstractConstraint,
    Inequality,
    Equality,
    StageConstraint,
    StateConstraint,
    ControlConstraint,
    CoupledConstraint,
    ConstraintParams,
	ConVal,
	ConstraintBlock

# constraint methods
export
    evaluate,
    jacobian,
    evaluate!,
    jacobian!,
    evaluate!,
    update_active_set!,
    max_violation,
    findmax_violation,
    is_bound,
    upper_bound,
    lower_bound,
	states,
	controls

# implemented constraints
export
	DynamicsConstraint,
	GoalConstraint,
	BoundConstraint,
	CircleConstraint,
	SphereConstraint,
	NormConstraint,
	LinearConstraint,
	CollisionConstraint,
	IndexedConstraint

# constraint sets
export
    ConstraintSet,
	ALConstraintSet,
	# ConstraintVals,
	ConstraintList,
    add_constraint!,
	num_constraints

include("trajectories.jl")
include("expansions.jl")
include("costfunctions.jl")
include("objective.jl")

include("abstract_constraint.jl")
include("constraints.jl")
include("dynamics_constraints.jl")
include("integration.jl")

include("cost.jl")
include("convals.jl")
# include("constraint_vals.jl")
# include("constraint_sets.jl")
# include("constraint_block.jl")
include("problem.jl")
include("conset.jl")

include("utils.jl")
end # module
