using TrajOptCore
using StaticArrays
using RobotDynamics
using BenchmarkTools
using LinearAlgebra
using ForwardDiff
using UnsafeArrays
using RobotZoo
using Test

import TrajOptCore: gen_jacobian, ConVal, ALConstraintSet

n,m = 13,4
x = @SVector rand(n)
u = @SVector rand(m)
z = KnotPoint(x,u,0.1)

# Goal Constraint
xf = @SVector rand(n)
con = GoalConstraint(xf)
size(con) == (n,n)
evaluate(con, z) == x - xf
∇c = gen_jacobian(con)
jacobian!(∇c, con, z)
∇c == I(n)

inds = 1:3
con = GoalConstraint(xf, inds)
size(con) == (3,n)
evaluate(con, z) == (x-xf)[1:3]
∇c = gen_jacobian(con)
jacobian!(∇c, con, z)
∇c == Matrix(I,3,n)

# Linear Constraint
p = 7
A = rand(p,n+m)
b = rand(p)
con = LinearConstraint(n,m,A,b,Inequality())
size(con) == (p,n+m)
evaluate(con, z) == A*[x;u] - b
∇c = gen_jacobian(con)
jacobian!(∇c, con, z)
∇c == A
con.A isa StaticMatrix
con.b isa StaticVector

# Circle Constraint
p = 4
xc = @SVector rand(p)
yc = @SVector rand(p)
rad = @SVector rand(p)
con = CircleConstraint(n, xc, yc, rad)
size(con) == (p,n)
evaluate(con, z) ≈ @. rad^2 - (xc - x[1])^2 - (yc - x[2])^2
∇c = gen_jacobian(con)
jacobian!(∇c, con, z)
sum(∇c[:,3:end]) ≈ 0

∇c ≈ ForwardDiff.jacobian(x->evaluate(con,x), x)
# @btime ForwardDiff.jacobian(x->evaluate($con,x), $x)


# Sphere Constraint
p = 5
xc = @SVector rand(p)
yc = @SVector rand(p)
zc = @SVector rand(p)
rad = @SVector rand(p)
con = SphereConstraint(n, xc, yc, zc, rad)
size(con) == (p,n)
evaluate(con, z) ≈ @. rad^2 - (xc - x[1])^2 - (yc - x[2])^2 - (zc - x[3])^2
∇c = gen_jacobian(con)
jacobian!(∇c, con, z)
sum(∇c[:,4:end]) ≈ 0
∇c ≈ ForwardDiff.jacobian(x->evaluate(con,x), x)

# Collision Constraint
x1 = @SVector [1,2,3]
x2 = x1 .+ 7
rad = rand()
con = CollisionConstraint(n, x1, x2, rad)
size(con) == (1,n)
evaluate(con, z)[1] ≈ rad^2 - norm(x[x1] - x[x2])^2
∇c = gen_jacobian(con)
jacobian!(∇c, con, z)
∇c ≈ ForwardDiff.jacobian(x->evaluate(con,x), x)

# State Norm
con = NormConstraint(n, m, 1.0, Inequality(), :state)
size(con) == (1,n+m)
evaluate(con, z)[1] ≈ x'x - 1
∇c = gen_jacobian(con)
jacobian!(∇c, con, z)
∇c[:] ≈ [2*x; zeros(m)]
∇c ≈ ForwardDiff.jacobian(x->evaluate(con,StaticKnotPoint(z,x)), z.z)

# Bounds Constraint
xmin = fill(-Inf,n)
xmax = fill(Inf,n)
xmin[1:3] .= -10
xmax[1:3] .= 10
umin = -4
umax = 4
bnd = BoundConstraint(n,m, x_min=xmin, x_max=xmax, u_min=umin, u_max=umax)
size(bnd) == (14,n+m)
evaluate(bnd, z) ≈ filter(isfinite,[x .- xmax; u .- umax; xmin .- x; umin .- u])
∇c = gen_jacobian(bnd)
jacobian!(∇c, bnd, z)
B = Diagonal(@SVector ones(n+m))
active = isfinite.([xmax;fill(umax,m);xmin;fill(umin,m)])
∇c ≈ [B;-B][active,:]

# Indexed Constraint
n_, m_ = 14, n+m
x_ = push(x, rand())
u_ = [u; x ./ 10]
z_ = KnotPoint(x_, u_, z.dt)

icon = IndexedConstraint(n_, m_, bnd)
size(icon) == (length(bnd), n_ + m_)
evaluate(icon, z_) == evaluate(bnd, z)
∇c = gen_jacobian(icon)
∇c0 = gen_jacobian(bnd)
jacobian!(∇c, icon, z_) == jacobian!(∇c0, bnd, z)
∇c[:,1:n] ≈ ∇c0[:,1:n]
∇c0[:,n .+ (1:m)] ≈ ∇c[:,n_ .+ (1:m)]

icon = IndexedConstraint(n_, m_, con)
size(icon) == (length(con), n_+m_)
evaluate(icon, z_) == evaluate(con, z)
∇c = gen_jacobian(icon)
∇c0 = gen_jacobian(con)
jacobian!(∇c, icon, z_) == jacobian!(∇c0, con, z)
∇c0[1:n] ≈ ∇c[1:n]


# Constraint List
N = 11
cons = ConstraintList(n,m,N)
add_constraint!(cons, bnd, 1:N-1)
cons.constraints[1] === bnd
cons.inds[1] == 1:N-1
add_constraint!(cons, con, N)
cons.constraints[2] === con
cons.inds[2] == N:N

goal = GoalConstraint(xf)
add_constraint!(cons, goal, N, 1)
cons[1] === goal
cons.inds[1] == N:N
length(cons) == 3


# ConVals
model = RobotZoo.Quadrotor()
n,m = size(model)
n̄ = RobotDynamics.state_diff_size(model)
dt = 0.1
Z = [KnotPoint(rand(model)..., dt) for k = 1:N]
G = [RobotDynamics.state_diff_jacobian(model, state(z)) for z in Z]
G = SizedMatrix{n,n̄}.(G)

inds = 1:N-1
C,c = TrajOptCore.gen_convals(n,m, bnd, inds)
cval = TrajOptCore.ConVal(n,m,bnd,inds,C,c)
evaluate!(cval, Z)
jacobian!(cval, Z)
max_violation(cval)

# @btime evaluate!($cval, $Z)
# @btime jacobian!($cval, $Z)
# @btime max_violation($cval)

# Error expansion
C,c = TrajOptCore.gen_convals(n̄, m, bnd, inds)
size(C[1],2) == (n̄+m)
errval = ConVal(n̄,m,bnd,inds,C,c,true)
evaluate!(errval, Z)
@test_throws ErrorException jacobian!(errval, Z)

# Create linked cval
conval = ConVal(n, m, errval)
evaluate!(conval, Z)
jacobian!(conval, Z)
@test conval.vals === errval.vals
@test conval.vals[1] === errval.vals[1]
@test conval.jac[1] !== errval.jac[1]

error_expansion!(errval, conval, model, G)
@test all([errval.jac[i] ≈ [conval.∇x[1]*G[i] conval.∇u[i]] for i = length(inds)])

# Test a non-RigidBody model
model = RobotZoo.Cartpole()
n,m = size(model)
n̄ = RobotDynamics.state_diff_size(model)
Z = [KnotPoint(rand(model)..., dt) for k = 1:N]
G = [RobotDynamics.state_diff_jacobian(model, state(z)) for z in Z]
G = SizedMatrix{n,n̄}.(G)

bnd = BoundConstraint(n, m, u_min=-3, u_max=3, x_max=[Inf,0.5,Inf,Inf], x_min=[-Inf,-0.5,-Inf,-Inf])
C,c = TrajOptCore.gen_convals(n̄, m, bnd, inds)
size(C[1],2) == (n̄+m)
errval = ConVal(n̄,m,bnd,inds,C,c, model isa RigidBody)
evaluate!(errval, Z)
jacobian!(errval, Z)

conval = ConVal(n, m, errval)
evaluate!(conval, Z)
jacobian!(conval, Z)
@test conval.vals === errval.vals
@test conval.vals[1] === errval.vals[1]
@test conval.jac[1] === errval.jac[1]
@test conval === errval

error_expansion!(errval, conval, model, G)


# AL ConSet
model = RobotZoo.Quadrotor()
n,m = size(model)
n̄ = RobotDynamics.state_diff_size(model)
dt = 0.1
Z = [KnotPoint(rand(model)..., dt) for k = 1:N]
G = [RobotDynamics.state_diff_jacobian(model, state(z)) for z in Z]
G = SizedMatrix{n,n̄}.(G)

import TrajOptCore: ALConstraintSet, reset!
conSet = ALConstraintSet(cons, model)
conSet.convals
evaluate!(conSet, Z)
jacobian!(conSet, Z)
max_violation(conSet)
TrajOptCore.max_penalty!(conSet)
reset!(conSet)
TrajOptCore.dual_update!(conSet)

J = zeros(N)
TrajOptCore.cost!(J, conSet)
sum(J)

maximum.(conSet.μ[1])
@btime evaluate!($conSet, $Z)
@btime jacobian!($conSet, $Z)
@btime max_violation($conSet)
@btime TrajOptCore.max_penalty!($conSet)
@btime reset!($conSet)
λ_max = 1e8
con_params = TrajOptCore.ConstraintParams()
@btime TrajOptCore.dual_update!($conSet)
@btime TrajOptCore.penalty_update!($conSet)
@btime TrajOptCore.cost!($J, $conSet)
