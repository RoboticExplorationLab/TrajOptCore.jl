using TrajOptCore
using RobotZoo
using StaticArrays
using LinearAlgebra
using Dynamics
using Test
using DifferentialRotations
const TOC = TrajOptCore
include("cartpole_problem.jl")

@testset "Constraints" begin
    include("constraint_tests.jl")
    include("dynamics_constraints.jl")
end

@testset "Problem" begin
    include("problem.jl")
end

@testset "Costs" begin
    include("cost_tests.jl")
end
