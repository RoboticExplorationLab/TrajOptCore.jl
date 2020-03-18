using TrajOptCore
using RobotZoo
using BenchmarkTools
using LinearAlgebra
using Test

model = RobotZoo.DoubleIntegrator(2)
n,m = size(model)
N = 11
x,u = rand(model)
x2,u2 = rand(model)
z1 = KnotPoint(x,u,0.1)
z2 = KnotPoint(x2,u2,0.1)
x0 = rand(model)[1]
xf = rand(model)[1]

lin = LinearConstraint{Equality,State}(n,m,rand(n-m,n), rand(n-m))
dyn = DynamicsConstraint(model, N)
bnd = BoundConstraint(n,m, u_min=-1, u_max=1)
ctl = NormConstraint{Equality,Control}(m,1.0)
init = GoalConstraint(x0)
goal = GoalConstraint(xf)

cons = [lin, bnd, ctl]
block = ConstraintBlock(dyn, cons)
@test size(block.D2,1) == n
@test size(block.D1,1) == n
@test size(block.C[1],1) == length(lin)
evaluate!(block, z1, z2)
@test norm(block.y,Inf) > 0
jacobian!(block, z1, z2)
@test block.D2 == -Matrix(I,n,n+m)

# @btime evaluate!($block, $z1, $z2)
# @btime jacobian!($block, $z1, $z2)

conSet = ConstraintSet(n,m,N)
add_constraint!(conSet, init, 1:1)
add_constraint!(conSet, dyn, 1:N-1)
add_constraint!(conSet, lin, 1:N)
add_constraint!(conSet, ctl, 1:N-1)
add_constraint!(conSet, bnd, 1:N-1)
add_constraint!(conSet, goal, N:N)
blocks = TrajOptCore.build_constraint_blocks(conSet)
@test blocks[end].terminal
@test !blocks[1].terminal
