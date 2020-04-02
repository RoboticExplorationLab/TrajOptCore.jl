using TrajOptCore
using RobotZoo
using StaticArrays
using LinearAlgebra
using Test
using RobotDynamics
const TOC = TrajOptCore
using BenchmarkTools
import TrajOptCore: ConVal
include("cartpole_problem.jl")

model = RobotZoo.Cartpole()
prob = gen_cartpole_prob()
bnd = TOC.get_constraints(prob).constraints[1]
n,m = size(prob)
N = 11

conSet = ALConstraintSet(prob.constraints, prob.model)

N = prob.N
rollout!(prob)
Z = prob.Z
vals = [@SVector zeros(model.n) for k = 1:N-1]

∇c = [SizedMatrix{n,n+m}(zeros(n,n+m)) for k = 1:N-1, i = 1:2]
dyn_con = DynamicsConstraint{RK3}(model, N)
@test TOC.width(dyn_con) == 2(n+m)
evaluate!(vals, dyn_con, Z)
jacobian!(∇c, dyn_con, Z)
@test (@allocated evaluate!(vals, dyn_con, Z)) == 0
@test (@allocated jacobian!(∇c, dyn_con, Z)) == 0

# ConVal
D,d = TOC.gen_convals(n,m,dyn_con,1:N-1)
TOC.widths(dyn_con)
TOC.gen_jacobian(dyn_con)
conval = ConVal(n,m, dyn_con, 1:N-1, D, d)
evaluate!(conval, Z)
jacobian!(conval, Z)
TOC.max_violation!(conval)
maximum(conval.c_max)

@test (@allocated evaluate!(conval, Z)) == 0
@test (@allocated jacobian!(conval, Z)) == 0


∇c = [zeros(SizedMatrix{n,2n+2m}) for k = 1:N-1]
dyn_con = DynamicsConstraint{HermiteSimpson}(model, N)
@test TOC.width(dyn_con) == 2(n+m)
evaluate!(vals, dyn_con, Z)
@which jacobian!(∇c, dyn_con, Z)
@btime jacobian!($∇c, $dyn_con, $Z)
@test (@allocated evaluate!(vals, dyn_con, Z)) == 0
@test (@allocated jacobian!(∇c, dyn_con, Z)) == 0

D,d = TOC.gen_convals(n,m,dyn_con,1:N-1)
conval =  ConVal(n,m,dyn_con, 1:N-1, D, d)
evaluate!(conval, Z)
jacobian!(conval, Z)
@test (@allocated evaluate!(conval, Z)) == 0
@test (@allocated jacobian!(conval, Z)) == 0


# Test default
dyn_con = DynamicsConstraint(model, N)
@test integration(dyn_con) == RK3 == TOC.DEFAULT_Q
