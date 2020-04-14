using StaticArrays
using LinearAlgebra
using TrajOptCore
using BenchmarkTools
using Test

n,m = 10,5
x = @SVector rand(n)
u = @SVector rand(m)
z = KnotPoint(x,u,0.1)

Q = Diagonal(@SVector rand(n))
R = Diagonal(@SVector rand(m))
H = rand(m,n)
q = @SVector rand(n)
r = @SVector rand(m)
c = rand()
xf = @SVector rand(n)
uf = @SVector rand(m)

# Test constructors
qcost = QuadraticCost(Q,R)
@test qcost isa QuadraticCost{n,m,Float64,<:Diagonal,<:Diagonal}
@test TrajOptCore.is_diag(qcost)

qcost = QuadraticCost(Q,R,H=H)
@test qcost.H ≈ H
@test qcost.zeroH == false
@test TrajOptCore.is_diag(qcost) == false

qcost = QuadraticCost(Q,R,H,q,r,c)
@test qcost.H ≈ H
@test qcost.q ≈ q
@test qcost.zeroH == false
@test TrajOptCore.is_diag(qcost) == false

qcost = QuadraticCost(Q,R,H*0,q,r,c)
@test qcost.zeroH == true
@test TrajOptCore.is_diag(qcost) == true

qcost = QuadraticCost(Matrix(Q), Matrix(R))
@test TrajOptCore.is_diag(qcost) == false

qcost = QuadraticCost(Q,R,q=Vector(q),r=Vector(r))
@test TrajOptCore.is_diag(qcost) == true

dcost = DiagonalCost(Q,R)
@test TrajOptCore.is_diag(dcost) == true

dcost = DiagonalCost(Q,R)
@test TrajOptCore.is_diag(dcost) == true

dcost = DiagonalCost(Q,R,H,q,r,c)
@test dcost isa DiagonalCost

dcost = DiagonalCost(diag(Q), diag(R))
@test dcost.Q == Q
@test dcost.R == R

qcost = QuadraticCost(diag(Q), diag(R))
@test qcost.Q == Q
@test qcost.R == R

# Test LQRCost constructors
dcost = LQRCost(Q,R,xf)
@test dcost isa DiagonalCost
@test dcost.Q == Q
@test dcost.q == -Q*xf
@test dcost.c ≈ 0.5*xf'Q*xf

qcost = LQRCost(Matrix(Q), Matrix(R), xf)
@test qcost isa QuadraticCost
@test qcost.Q == Q
@test qcost.q == -Q*xf
@test qcost.c ≈ 0.5*xf'Q*xf

q2 = copy(qcost)
@test q2.Q !== qcost.Q
@test q2.q !== qcost.q

qcost = LQRCost(Q,R,xf)

E = QuadraticCost{Float64}(n,m)
@btime TrajOptCore.stage_cost($qcost, $x, $u)
@btime TrajOptCore.gradient!($E, $qcost, $x, $u)
@btime TrajOptCore.hessian!($E, $qcost, $x, $u)
