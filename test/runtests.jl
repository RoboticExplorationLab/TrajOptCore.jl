using TrajOptCore
using RobotZoo
using StaticArrays
using LinearAlgebra
using Dynamics
using Test
using DifferentialRotations
const TOC = TrajOptCore

@testset "Constraints" begin
    include("constraint_tests.jl")
end

@testset "Problem" begin
    include("problem.jl")
end
