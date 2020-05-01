using ForwardDiff
using DiffResults
using Rotations
using StaticArrays
using LinearAlgebra
using BenchmarkTools
using Zygote
using FiniteDiff
using FiniteDifferences

function dynamics(x::AbstractVector, u::AbstractVector)
      # x = X[1:3]
      q = normalize(UnitQuaternion(x[4],x[5],x[6],x[7]))
      v = view(x,8:10)
      ω = x[@SVector [11,12,13]]

      # Parameters
      m = 1.0
      J = Diagonal(SA[0.0023, 0.0023, 0.004])
      Jinv = Diagonal(@SVector ones(3))
      g = @SVector [0,0,-9.81]
      L = 0.1750
      kf = 1.0
      km = 0.0245

      w1 = u[1]
      w2 = u[2]
      w3 = u[3]
      w4 = u[4]

      F1 = kf*w1;
      F2 = kf*w2;
      F3 = kf*w3;
      F4 = kf*w4;
      F = @SVector [0., 0., F1+F2+F3+F4] #total rotor force in body frame

      M1 = km*w1;
      M2 = km*w2;
      M3 = km*w3;
      M4 = km*w4;
      tau = @SVector [L*(F2-F4), L*(F3-F1), (M1-M2+M3-M4)] #total rotor torque in body frame

      qdot = Rotations.kinematics(q, ω)
      vdot = g + (1/m)*(q*F) #acceleration in world frame
      omdot = Jinv*(tau - cross(ω,J*ω)) #Euler's equation: I*ω + ω x I*ω = constraint_decrease_ratio
      @SVector [v[1], v[2], v[3], qdot[1], qdot[2], qdot[3], qdot[4], vdot[1], vdot[2], vdot[3], omdot[1], omdot[2], omdot[3]]
end

function dynamics!(xdot, x, u)
    xdot .= dynamics(x,u)
end

x = [(@SVector rand(3)); normalize(@SVector rand(4)); (@SVector rand(6))]
u = @SVector rand(4)
xdot = @MVector zeros(13)

x2 = Vector(x)
u2 = Vector(u)
xdot2 = Vector(xdot)

z = [x;u]
zm = MVector(z)
z2 = [x2;u2]
f_aug(z) = dynamics(z[SVector{13}(1:13)], z[SA[14,15,16,17]])
f_aug!(xdot,z) = dynamics!(xdot,z[SVector{13}(1:13)], z[SA[14,15,16,17]])


#--- Benchmark the dynamics
bs = @benchmark dynamics($x,$u)
bs_ip = @benchmark dynamics!($xdot,$x,$u)
judge(median(bs), median(bs_ip))  # not in-place vs in place

bs2 = @benchmark dynamics($x2,$u2)
bs2_ip = @benchmark dynamics!($xdot2,$x2,$u2)
judge(median(bs), median(bs2))
judge(median(bs_ip), median(bs2_ip))

#== Summary
All are non-allocating, and dynamics in-place is only slightly slower
==#

#--- ForwardDiff Jacobian Benchmarks
fd  = @benchmark ForwardDiff.jacobian(f_aug, $z)
fd2 = @benchmark ForwardDiff.jacobian(f_aug, $z2)
judge(median(fd), median(fd2))  # Static arrays is much faster

∇c = zeros(13,17)
ip_fd  = @benchmark ForwardDiff.jacobian!($∇c, f_aug, $z)
ip2_fd = @benchmark ForwardDiff.jacobian!($∇c, f_aug, $z2)
judge(median(ip_fd), median(ip2_fd))  # Static arrays is much faster
judge(median(fd), median(ip_fd))  # in-place is slightly slower for static arrays
judge(median(fd2), median(ip2_fd))  # in-place is faster for dynamic arrays

ipp_fd  = @benchmark ForwardDiff.jacobian!($∇c, f_aug!, $xdot, $z)
ipp2_fd = @benchmark ForwardDiff.jacobian!($∇c, f_aug!, $xdot, $z2)
judge(median(ipp_fd), median(fd))  # much slower than not in place for static arrays
judge(median(ipp2_fd), median(ip2_fd))  # slower than in place for dynamic arrays
judge(median(ipp2_fd), median(fd2))  # slower than in place for dynamic arrays

#== Summary
Static Arrays makes the Jacobians 2x faster and non-allocating
For static arrays:
    not in place < in place < in place with in place vector
For dynamic arrays
    in place < not in place <= in place with in place vector
==#

#--- DiffResults
res = DiffResults.JacobianResult(x,[x;u])
res2 = DiffResults.JacobianResult(x2,[x2;u2])
res = jacobian!(res, x, u)
jacobian!(res2, x2, u2)
DiffResults.jacobian(res2) ≈ ∇c
DiffResults.jacobian(res) ≈ ∇c

dr = @benchmark $res = jacobian!($res, $x, $u)
dr_ = @benchmark jacobian!($res2, $x, $u)  # best
dr2 = @benchmark $res2 = jacobian!($res2, $x2, $u2)
dr2_ip = @benchmark jacobian!($∇c, $xdot, $x2, $u2, $s)
judge(median(dr), median(dr_))  # mutable res is better
judge(median(dr), median(dr2))  # static arrays is better
judge(median(dr_), median(fd))  # better to not use diff results  with static arrays
judge(median(dr2_ip), median(dr2)) # faster to use diff results with dynamics arrays

#== Summary
For static arrays
    better to not use DiffResults
For dynamics arrays
    better to use DiffResults (but uses more memory?)
==#


#--- Zygote
λ = @SVector rand(13)
y,back = pullback(dynamics, x, u)
y ≈ dynamics(x,u)
back(λ)

#== Summary
Doesn't work?
==#

#--- FiniteDiff
∇c2 = zeros(13,17)
cache = FiniteDiff.JacobianCache(z,xdot,zero(xdot))
cache2 = FiniteDiff.JacobianCache(z2,x2,copy(x2))
norm(ForwardDiff.jacobian(f_aug, z) - FiniteDiff.finite_difference_jacobian(f_aug, z)) < 1e-6
FiniteDiff.finite_difference_jacobian!(∇c2, f_aug!, z2)
norm(∇c - ∇c2) < 1e-6
norm(FiniteDiff.finite_difference_jacobian(f_aug, z, cache) - ∇c) < 1e-6
FiniteDiff.finite_difference_jacobian!(∇c2, f_aug!, z2, cache2)

@btime FiniteDiff.finite_difference_jacobian(f_aug, $z)
@btime FiniteDiff.finite_difference_jacobian(f_aug, $z, $cache)
@btime FiniteDiff.finite_difference_jacobian!($∇c2, $f_aug!, $zm)
@btime FiniteDiff.finite_difference_jacobian!($∇c2, $f_aug!, $z2)
@btime FiniteDiff.finite_difference_jacobian!($∇c2, $f_aug!, $z2, $cache2)
@btime FiniteDiff.finite_difference_jacobian!($∇c2, $f_aug!, $zm, $cache2)

#== Summary
Dynamic arrays with cache are the fastest, but still slower than ForwardDiff with static arrays
==#

#--- Finite Differences
FiniteDifferences.jacobian(central_fdm(2,1), f_aug, z)[1] ≈ ∇c
FiniteDifferences.jacobian(central_fdm(2,1), f_aug, z2)[1] ≈ ∇c

fdm = central_fdm(2,1)
zs = SizedVector{17}(z2)
@btime FiniteDifferences.jacobian($fdm, $f_aug, $z)
@btime FiniteDifferences.jacobian($fdm, $f_aug, $zm) # slowest
@btime FiniteDifferences.jacobian($fdm, $f_aug, $zs) # better than SVector and MVector
@btime FiniteDifferences.jacobian($fdm, $f_aug, $z2) # faster and less allocations

fdm = forward_fdm(2,1)
@btime FiniteDifferences.jacobian($fdm, $f_aug, $z)
@btime FiniteDifferences.jacobian($fdm, $f_aug, $zm) # slowest
@btime FiniteDifferences.jacobian($fdm, $f_aug, $zs) # better than SVector and MVector
@btime FiniteDifferences.jacobian($fdm, $f_aug, $z2) # faster and less allocations

#== Summary
Much slower than the other methods, but get the same accuracy as ForwardDiff
Still fastest to use dynamic arrays
==#


#--- Jacobian-vector Products
λ = @SVector rand(13)
jv = ForwardDiff.gradient(z->f_aug(z)'*λ, z)

fd_in = @benchmark ForwardDiff.gradient(z->f_aug(z)'*$λ, $z)
fd_out = @benchmark ForwardDiff.jacobian(f_aug, $z)'*$λ
judge(median(fd_in), median(fd_out))  # no difference

norm(FiniteDiff.finite_difference_gradient(z->f_aug(z)'λ,z2) - jv)
fin_diff = @benchmark FiniteDiff.finite_difference_gradient(z->f_aug(z)'*$λ,$z2)
judge(median(fd_in), median(fin_diff))

norm(FiniteDifferences.grad(fdm, z->f_aug(z)'λ, z2)[1] - jv)
@btime FiniteDifferences.grad($fdm, z->f_aug(z)'*$λ, $z2)
norm(FiniteDifferences.j′vp(fdm, f_aug, λ, z2)[1] - jv)
@btime FiniteDifferences.j′vp($fdm, $f_aug, $λ, $z2)

#--- Jacobian-vector Product w/ Reverse Diff
using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile

#== Summary
ReverseDiff doesn't appear to work with StaticArrays
==#

#--- Hessian-vector Products
function hessvec(z,λ)
    ForwardDiff.hessian(z->f_aug(z)'*λ, z)
end
function jacjacvec(z,λ)
    ForwardDiff.jacobian(z->ForwardDiff.jacobian(f_aug, z)'*λ, z)
end
hess_vec = @benchmark hessvec($z,$λ)
jacjac_vec = @benchmark jacjacvec($z,$λ)
judge(median(hess_vec), median(jacjac_vec))

hv = hessvec(z,λ)

∇cy = zeros(17,17)
cache = FiniteDiff.HessianCache(z)
norm(FiniteDiff.finite_difference_hessian(z->f_aug(z)'λ, z) - hv)
norm(FiniteDiff.finite_difference_hessian!(∇cy, z->f_aug(z)'λ, z) - hv)
norm(FiniteDiff.finite_difference_hessian!(∇cy, z->f_aug(z)'λ, z, cache) - hv)

@btime FiniteDiff.finite_difference_hessian(z->f_aug(z)'*$λ, $z)
jacvec_findiff = @benchmark FiniteDiff.finite_difference_hessian!($∇cy, z->f_aug(z)'*$λ, $z)
judge(median(jacvec_findiff), median(jacjac_vec))


function jacvec(f, x, p, eps=1e-6)
    (f(x + eps*p) - f(x))/eps
end
Jx = ForwardDiff.jacobian(f_aug, z)*z
norm(jacvec(f_aug, z, z) - Jx)
@btime jacvec(f_aug, $z, $z)
@btime ForwardDiff.jacobian(f_aug, $z)*$z
@btime FiniteDifferences.jvp($fdm, f_aug, ($z, $z))

function hessvec(f, x, p, eps=1e-6)
    f(x + eps*)
end




#== OVERALL SUMMARAY

For Jacobians
=##
