using TrajOptCore
using BenchmarkTools
using TrajectoryOptimization
using ForwardDiff
using Test
const TO = TrajectoryOptimization

include("../src/nlp.jl")

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

# Test second-order constraint term
NN = N*n + (N-1)*m
P = sum(num_constraints(prob))
data = (D=spzeros(P,NN), d=zeros(P), G=spzeros(NN,NN), λ=zeros(P))
conSet = NLPConstraintSet(prob.model, prob.constraints, data)
prob.Z[2].z += rand(n+m)
conSet.λ[5][1] .= rand(n)
evaluate!(conSet, prob.Z)
@test conSet.convals[end].vals[1] != zeros(n)
@test conSet.convals[end].vals[3] == zeros(n)
∇jacobian!(conSet.hess, conSet, prob.Z, conSet.λ)
@test conSet.hess[end][1] != zeros(n+m,n+m)
@test conSet.hess[end][2] == zeros(n+m,n+m)

# Build NLP
nlp = TrajOptNLP(prob)
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

G = ForwardDiff.hessian(x->eval_f(nlp,x), Z)
G0 = nlp.data.G
G0 .*= 0
@test hess_f!(nlp, Z) ≈ G
@test nlp.E.cost[1].Q.parent === G0
@btime hess_f!($nlp, $Z)
@btime ForwardDiff.hessian(x->eval_f($nlp,x), $Z)

# Constraint Functions
evaluate!(nlp.conSet, prob.Z)
c_max = max_violation(nlp.conSet)
c = eval_c!(nlp, Z)
@test max_violation(nlp.conSet) ≈ c_max
@btime eval_c!($nlp, $Z)
@btime evaluate!($nlp.conSet, $(prob.Z))

jac_c!(nlp, Z)
@btime jac_c!($nlp, $Z)

nlp.conSet.λ[end][1] .= 0
@test hess_L!(nlp, Z) ≈ G
nlp.conSet.λ[end][1] .= rand(n)
@test !(hess_L!(nlp, Z) ≈ G)


# Test cost hessian structure
@test nlp.obj isa Objective{<:DiagonalCostFunction}
G_ = hess_f_structure(nlp)
@test nnz(G_) == NN
@test diag(G_) == 1:NN

obj_ = QuadraticObjective(n,m,N)
prob_ = Problem(prob, obj=obj_)
nlp_ = TrajOptNLP(prob_)
@test !(nlp_.obj isa Objective{<:DiagonalCostFunction})
G_ = hess_f_structure(nlp_)
@test nnz(G_) == (N-1)*(n+m)^2 + n^2
@test G_[1:n+m, 1:n+m] == reshape(1:(n+m)^2, n+m, n+m)

r,c = get_rc(G_)
@test [G_[r[i], c[i]] for i = 1:nnz(G_)] == 1:nnz(G_)


# Test jacobian structure
D_ = jac_structure(nlp)
nlp.conSet.convals[1].jac[1] == reshape(1:n^2, n,n)
nlp.conSet.convals[2].jac[1] == reshape(9 .+ (1:(n+m)*4), 4, n+m)
nlp.data.D .= 0
jac_c!(nlp, Z)
@test D_[end,end] == nnz(nlp.data.D)

# Constraint type
IE = constraint_type(nlp)
@test IE[1:n] == ones(n)             # initial state constraint
@test IE[n .+ (1:4)] == zeros(4)     # bound constraint
@test IE[(n+4) .+ (1:n)] == ones(n)  # dynamics constraint

# Test bounds removal
cons = prob.constraints
zL = fill(-Inf,NN)
zU = fill(+Inf,NN)
primal_bounds!(zL, zU, cons, false)
@test zL[1:n] == zeros(n)
@test zU[1:n] == zeros(n)
@test zL[n+1:n+m] == fill(-2,m)
@test zU[n+m+1:2n+m] == [0.25, 1.501, Inf]
@test zL[n+m+1:2n+m] == [-0.25, -0.001, -Inf]
@test zL[end-n+1:end] == zeros(n)
@test zU[end-n+1:end] == zeros(n)

@test length(cons) == 5
cons2 = copy(cons)
primal_bounds!(zL, zU, cons2, true)
@test length(cons2) == 2
@test length(cons) == 5
