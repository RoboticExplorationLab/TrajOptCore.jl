using TrajOptCore
using StaticArrays
using LinearAlgebra
using RobotDynamics
using RobotZoo
using Rotations
using BenchmarkTools
using Test

model = RobotZoo.Quadrotor()
n,m,N = size(model)..., 11
n̄ = RobotDynamics.state_diff_size(model)
dt = 0.1
Z = [KnotPoint(rand(model)..., dt) for k = 1:N]
G = [RobotDynamics.state_diff_jacobian(model, state(z)) for z in Z]
G = SizedMatrix{n,n̄}.(G)
push!(G, zero(G[end]))

# Build cost
Q = Diagonal(@SVector fill(1.0,n))
R = Diagonal(@SVector fill(0.1,m))
x0 = zeros(model)[1]
xf = RobotDynamics.build_state(model, ones(3), UnitQuaternion(I), zeros(3), zeros(3))
obj = LQRObjective(Q,R,10Q,xf,N)

Jexp = QuadraticObjective(n,m,N)

cost_gradient!(Jexp, obj, Z)
@test !any(Jexp.const_grad)
@test !any(Jexp.const_hess)
cost_hessian!(Jexp, obj, Z)
@test all(Jexp.const_hess)

# @btime cost_gradient!($Jexp, $obj, $Z)
# @btime cost_hessian!($Jexp, $obj, $Z, true)
# @btime cost_hessian!($Jexp, $obj, $Z, false)

# Error expansion
E = QuadraticObjective(n̄,m,N)
@test E[1] isa QuadraticCost{n̄,m,<:Any,<:SizedMatrix{n̄,n̄},<:SizedMatrix{m,m}}
@test E[1].Q !== E[2].Q !== E[N].Q

Jexp = QuadraticObjective(E, model)
@test Jexp[1].Q !== E[1].Q
@test Jexp[1].R === E[1].R
@test Jexp[1].q !== E[1].q
@test Jexp[1].r === E[1].r

@test size(Jexp[1].Q) == (n,n)
@test size(E[1].Q) == (n̄,n̄)

cost_gradient!(Jexp, obj, Z)
cost_hessian!(Jexp, obj, Z)

Jexp[1].Q
error_expansion!(E, Jexp, model, Z, G)
RobotDynamics.∇²differential(model, state(Z[1]), Jexp[1].q)

@test E[1].Q[1:3,1:3] ≈ Jexp[1].Q[1:3,1:3]
@test E[1].Q[7:12,7:12] ≈ Jexp[1].Q[8:13,8:13]
@test E[1].q ≈ G[1]'Jexp[1].q
# @btime error_expansion!($E, $Jexp, $model, $Z, $G)
