using TrajOptCore
using BenchmarkTools
using StaticArrays
using LinearAlgebra
using RobotDynamics
using RobotZoo

import TrajOptCore: Jacobian, ConVal, ConSet

# Test ConVals
model = RobotZoo.Quadrotor()
n0,m = size(model)
n = RobotDynamics.state_diff_size(model)
N = 11
dt = 0.1
Z = [KnotPoint(rand(model)...,dt) for k = 1:N]
G = [RobotDynamics.state_diff_jacobian(model, state(z)) for z in Z]
G = SizedMatrix{n0,n}.(G)

A = rand(m,n0)
b = rand(m)
con = LinearConstraint{Equality,State}(n0,m,A,b)
inds = 1:N
vals = [SizedVector{m}(zeros(m)) for k in inds]
jac = [Jacobian(SizedMatrix{m,n}(zeros(m,n)), G[k], true) for k in inds]
[j.rmul for j in jac]
TrajOptCore.width(con)

cval = ConVal(con, inds, vals, jac)
[j.lmul for j in cval.jac]

evaluate!(cval, Z)
@btime evaluate!($cval, $Z)
jacobian!(cval, Z)
@btime jacobian!($cval, $Z)
error_expansion!(cval)
@btime error_expansion!($cval)
TrajOptCore.max_violation!(cval)
max_violation(cval) ≈ norm(norm.(vals,Inf),Inf)
@btime TrajOptCore.max_violation($cval)

conSet

# Test Jacobian
n0, n = 13,12
N = 11
F = [SizedMatrix{n,n}(zeros(n,n)) for k = 1:N]
G = [SizedMatrix{n0,n}(rand(n0,n)) for k = 1:N]

# Double-sided
jac = Jacobian(F[1], G[1], true, G[1], true)
@test sum(jac.F) == 0
F[1] .= 1
@test sum(jac.F) == n*n
@test jac.F === F[1]
@test size(jac.∇f) == (n0,n0)

@test G[1] === jac.G
@test G[1] === jac.Gt.parent

jac.∇f .= 1
error_expansion!(jac)
@test jac.F ≈ G[1]'jac.∇f*G[1]

@btime error_expansion!($jac)

# Double-sided, different (e.g. dynamics constraints)
jac = Jacobian(F[1], G[1], true, G[2], true)
@test G[1] === jac.G
@test G[2] === jac.Gt.parent
@test jac.G !== jac.Gt.parent
jac.∇f .= 2
error_expansion!(jac)
@test jac.F ≈ G[2]'jac.∇f*G[1]

@btime error_expansion!($jac)

# Right-sided (e.g. constraint)
p = 6
F = [SizedMatrix{p,n}(zeros(p,n)) for k = 1:N]
jac = Jacobian(F[1], G[1], true)
@test jac.F === F[1]
@test jac.G === G[1]
@test_throws UndefRefError jac.Gt
jac.∇f .= 3
error_expansion!(jac)
@test jac.F ≈ jac.∇f*G[1]

@btime error_expansion!($jac)

# Left-sided vector (e.g. cost gradient)
F = [SizedVector{n}(zeros(n)) for k = 1:N]
jac = Jacobian(F[1], G[1], false, G[1], true)
@test jac.∇f isa AbstractVector
@test jac.F === F[1]
@test jac.Gt.parent === G[1]
jac.∇f .= 4
error_expansion!(jac)
@test jac.F ≈ G[1]'jac.∇f

@btime error_expansion!($jac)

# No expansion
@test_throws AssertionError jac = Jacobian(F[1], G[1], false, G[1], false)
G = [SizedMatrix{n0,n0}(zeros(n0,n0)) for k = 1:N]
jac = Jacobian(F[1], G[1], false, G[1], false)
@test jac.F === F[1]
@test jac.∇f === F[1]

@btime error_expansion!($jac)
