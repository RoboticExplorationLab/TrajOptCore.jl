using StaticArrays
using LinearAlgebra
using TrajOptCore
using BenchmarkTools
using Test

n,m = 10,5
x = @SVector rand(n)
u = @SVector rand(m)
z = KnotPoint(x,u,0.1)


# Test hessian inverse
Q = Diagonal(@SVector rand(n))
R = Diagonal(@SVector rand(m))
costfun = QuadraticCost(Q,R)
J = Diagonal([diag(Q); diag(R)])
z_ = costfun\z
@test J\z.z ≈ z_.z

costfun = DiagonalCost(Q,R)
J = Diagonal([diag(Q); diag(R)])
z_ = costfun\z
@test J\z.z ≈ z_.z

Q = rand(n,n)
R = rand(m,m)
Q = SizedMatrix{n,n}(Q'Q)
R = SizedMatrix{m,m}(R'R)
costfun = QuadraticCost(Q,R)
@test costfun.zeroH
@test_throws UndefRefError costfun.Sinv
J = cat(Q,R,dims=[1,2])
z_ = costfun\z
@test J\z.z ≈ z_.z

H = SizedMatrix{n,m}(rand(n,m))
costfun = QuadraticCost(Q,R,H=H)
@test !costfun.zeroH
J = [Q H; H' R]
z_ = costfun\z
@test z_.z ≈ J\z.z
@btime $costfun\$z
