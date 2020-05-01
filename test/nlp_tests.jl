using TrajOptCore
using BenchmarkTools
using TrajectoryOptimization
using ForwardDiff
using Test
const TO = TrajectoryOptimization

# Test NLPTraj iteration
n,m,N = 3,2,101
NN = N*n + (N-1)*m
Z0 = Traj(n,m,0.1,N)
Zdata = TrajData(Z0)
Z = rand(NN)
Z_ = NLPTraj(Z,Zdata)
@test Z_[1] isa StaticKnotPoint
@test state(Z_[2]) == Z[(n+m) .+ (1:n)]
@test state(Z_[end]) == Z[end-n+1:end]
@test length(Z_) == N
@test size(Z_) == (N,)
@test [z for z in Z_] isa Vector{<:StaticKnotPoint}
@test eltype(Z_) == StaticKnotPoint{n,m,Float64,n+m}

# Test with problem
prob = Problems.DubinsCar(:parallel_park)[1]
add_dynamics_constraints!(prob)
n,m,N = size(prob)
cons = prob.constraints

nlp = TrajOptNLP(prob, slacks=true)
Zdata = nlp.Z.Zdata

Z = Primals(prob).Z
Z_ = NLPTraj(Z, Zdata);
z = StaticKnotPoint(Z,Zdata,1)
@test state(z) ≈ state(prob.Z[1])

# Cost functions
cost(prob) ≈ eval_f(nlp, Z)

grad_f!(nlp, Z) ≈ ForwardDiff.gradient(x->eval_f(nlp,x), Z)
@btime grad_f!($nlp, $Z)
@btime ForwardDiff.gradient(x->eval_f($nlp,x), $Z)

hess_f!(nlp, Z) ≈ ForwardDiff.hessian(x->eval_f(nlp,x), Z)
@btime hess_f!($nlp, $Z)
@btime ForwardDiff.hessian(x->eval_f($nlp,x), $Z)

# Constraint Functions
evaluate!(nlp.conSet, prob.Z)
c_max = max_violation(nlp.conSet)
c = eval_c!(nlp, Z)
@test max_violation(nlp.conSet) ≈ c_max
@btime eval_c!($nlp, $Z)
@btime evaluate!($nlp.conSet, $(prob.Z))

nlp.conSet.cinds

jac_c!(nlp, Z)
@btime jac_c!($nlp, $Z)
