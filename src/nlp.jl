using SparseArrays

mutable struct NLPData{T}
	G::SparseMatrixCSC{T,Int}
	g::Vector{T}
	D::SparseMatrixCSC{T,Int}
	d::Vector{T}
	λ::Vector{T}
	zL::Vector{T}  # primal lower bounds
	zU::Vector{T}  # primal upper bounds
end

function NLPData(NN::Int,P::Int)
	G = spzeros(NN,NN)
	g = zeros(NN)
	D = spzeros(P,NN)
	d = zeros(P)
	λ = zeros(P)
	zL = zeros(NN)
	zU = zeros(NN)
	NLPData(G,g,D,d,λ,zL,zU)
end

"""
	NLPConstraintSet{T}

Constraint set that updates views to the NLP constraint vector and Jacobian.
"""
struct NLPConstraintSet{T} <: AbstractConstraintSet
    convals::Vector{ConVal}
    errvals::Vector{ConVal}
	cinds::Vector{Vector{UnitRange{Int}}}
	λ::Vector{Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}}
	hess::Vector{Matrix{SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}}}
	c_max::Vector{T}
end

function NLPConstraintSet(model::AbstractModel, cons::ConstraintList, data,
		jac_structure=:by_knotpoint; slacks::Bool=false)
	if !has_dynamics_constraint(cons)
		throw(ArgumentError("must contain a dynamics constraint"))
	end

	n,m = size(model)
	n̄ = RobotDynamics.state_diff_size(model)
	ncon = length(cons)
	N = length(cons.p)

	zinds = [(k-1)*(n+m) .+ (1:n+m) for k = 1:N-1]
	push!(zinds, (N-1)*(n+m) .+ (1:n))

	# Block sizes
	NN = N*n̄ + (N-1)*m
	P = sum(num_constraints(cons))

	# Initialize arrays
	D = data.D
	d = data.d

	# Create ConVals as views into D and d
	cinds = gen_con_inds(cons, jac_structure)
	zinds = [(k-1)*(n+m) .+ (1:n+m) for k = 1:N]
	useG = model isa LieGroupModel
	errvals = map(enumerate(zip(cons))) do (i,(inds,con))
		C,c = TrajOptCore.gen_convals(D, d, cinds[i], zinds, con, inds)
		ConVal(n̄, m, con, inds, C, c)
	end
	convals = map(errvals) do errval
		ConVal(n, m, errval)
	end
	errvals = convert(Vector{ConVal}, errvals)
	convals = convert(Vector{ConVal}, convals)

	# Create views into the multipliers
	λ = map(1:ncon) do i
		map(cinds[i]) do ind
			view(data.λ, ind)
		end
	end

	# Create views into the Hessian matrix
	G = data.G
	hess1 = map(zip(cons)) do (inds,con)
		zind = get_inds(con, n̄, m)[1]
		map(enumerate(inds)) do (i,k)
			zind_ = zind .+ ((k-1)*(n+m))
			view(G, zind_, zind_)
		end
	end
	hess2 = map(zip(cons)) do (inds,con)
		zind = get_inds(con, n̄, m)
		map(enumerate(inds)) do (i,k)
			if length(zind) > 1
				zind_ = zind[2] .+ ((k-1)*(n+m))
			else
				zind_ = (1:0) .+ ((k-1)*(n+m))
			end
			view(G, zind_, zind_)
		end
	end
	hess = map(zip(hess1, hess2)) do (h1,h2)
		[h1 h2]
	end

	NLPConstraintSet(convals, errvals, cinds, λ, hess, zeros(ncon))
end

@inline get_convals(conSet::NLPConstraintSet) = conSet.convals
@inline get_errvals(conSet::NLPConstraintSet) = conSet.errvals

function norm_violation(conSet::NLPConstraintSet, p=2)
	norm(conSet.d, p)
end

@inline ∇jacobian!(conSet::NLPConstraintSet, Z) = ∇jacobian!(conSet.hess, conSet, Z, conSet.λ)

"""
	gen_con_inds(cons::ConstraintList, structure::Symbol)

Generate the indices into the concatenated constraint vector for each constraint.
Determines the bandedness of the Jacobian
"""
function gen_con_inds(conSet::ConstraintList, structure=:by_knotpoint)
	n,m = conSet.n, conSet.m
    N = length(conSet.p)
    numcon = length(conSet.constraints)
    conLen = length.(conSet.constraints)

    # cons = [[@SVector ones(Int,length(con)) for j in eachindex(conSet.inds[i])]
	# 	for (i,con) in enumerate(conSet.constraints)]
	cons = [[1:0 for j in eachindex(conSet.inds[i])] for i in 1:length(conSet)]

    # Dynamics and general constraints
    idx = 0
	if structure == :by_constraint
	    for (i,con) in enumerate(conSet.constraints)
			for (j,k) in enumerate(conSet.inds[i])
				cons[i][TrajOptCore._index(con,k)] = idx .+ (1:conLen[i])
				idx += conLen[i]
	        end
	    end
	elseif structure == :by_knotpoint
		for k = 1:N
			for (i,con) in enumerate(conSet.constraints)
				inds = conSet.inds[i]
				if k in inds
					j = k -  inds[1] + 1
					cons[i][j] = idx .+ (1:conLen[i])
					idx += conLen[i]
				end
			end
		end
	elseif structure == :by_block
		sort!(conSet)  # WARNING: may modify the input
		idx = zeros(Int,N)
		for k = 1:N
			for (i,(inds,con)) in enumerate(zip(conSet))
				if k ∈ inds
					j = k - inds[1] + 1
					cons[i][j] = idx[k] .+ (1:length(con))
					idx[k] += length(con)
				end
			end
		end
	end
    return cons
end

"""
	QuadraticViewCost{n,m,T}

A quadratic cost that is a view into a large sparse matrix
"""
struct QuadraticViewCost{n,m,T} <: QuadraticCostFunction{n,m,T}
	Q::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
	R::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
	H::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
	q::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
	r::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
	c::T
	zeroH::Bool
	terminal::Bool
	function QuadraticViewCost(Q::SubArray, R::SubArray, H::SubArray,
		q::SubArray, r::SubArray, c::Real; checks::Bool=true, terminal::Bool=false)
		if checks
			TrajOptCore.run_posdef_checks(Q,R)
		end
		n,m = length(q), length(r)
        T = promote_type(eltype(Q), eltype(R), eltype(H), eltype(q), eltype(r), typeof(c))
        zeroH = norm(H,Inf) ≈ 0
		new{n,m,T}(Q, R, H, q, r, c, zeroH, terminal)
	end
end

function QuadraticViewCost(G::SparseMatrixCSC, g::Vector,
		cost::TrajOptCore.QuadraticCostFunction, k::Int)
	n,m = state_dim(cost), control_dim(cost)
	ix = (k-1)*(n+m) .+ (1:n)
	iu = ((k-1)*(n+m) + n) .+ (1:m)
	NN = length(g)

	Q = view(G,ix,ix)
	q = view(g,ix)

	if cost.Q isa Diagonal
		for i = 1:n; Q[i,i] = cost.Q[i,i] end
	else
		Q .= cost.Q
	end
	q .= cost.q

	# Point the control-dependent values to null matrices at the terminal time step
	if cost.terminal &&  NN == k*n + (k-1)*m
		R = view(spzeros(m,m), 1:m, 1:m)
		H = view(spzeros(m,n), 1:m, 1:n)
		r = view(zeros(m), 1:m)
	else
		R = view(G,iu,iu)
		H = view(G,iu,ix)
		r = view(g,iu)
		if cost.R isa Diagonal
			for i = 1:m; R[i,i] = cost.R[i,i] end
		else
			R .= cost.R
		end
		r .= cost.r
		if !TrajOptCore.is_blockdiag(cost)
			H .= cost.H
		end
	end

	QuadraticViewCost(Q, R, H, q, r, cost.c, checks=false, terminal=cost.terminal)
end

TrajOptCore.is_blockdiag(cost::QuadraticViewCost) = cost.zeroH

"""
	ViewKnotPoint{T,n,m}
"""
struct ViewKnotPoint{T,N,M} <: AbstractKnotPoint{T,N,M}
    z::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    dt::T
    t::T
    function ViewKnotPoint(z::SubArray, _x::SVector{N,Int}, _u::SVector{M,Int},
            dt::T1, t::T2) where {N,M,T1,T2}
        T = promote_type(T1,T2)
        new{T,N,M}(z, _x, _u, dt, t)
    end
end

function ViewKnotPoint(z::SubArray, n, m, dt, t=0.0)
    ix = SVector{n}(1:n)
    iu = SVector{m}(n .+ (1:m))
    ViewKnotPoint(z, ix, iu, dt, t)
end

"""
	Primals(n,m,T)

Storage container for a state-control trajectory
"""
struct Primals{n,m,T}
    Z::Vector{T}
    Z_::Vector{ViewKnotPoint{T,n,m}}
    X::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}
    U::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}
    X_::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    U_::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
end

function Primals(n,m,N,tf)
    dt = tf/(N-1)
    NN = N*n + (N-1)*m
    Z = zeros(NN)
    ix, iu, iz = 1:n, n .+ (1:m), 1:n+m
    iX = [ix .+ k*(n+m) for k = 0:N-1]
    iU = [iu .+ k*(n+m) for k = 0:N-2]

    tf = (N-1)*dt
    t = range(0,tf,length=N)
    dt = t[2]
    Z_ = [ViewKnotPoint(view(Z,iz .+ k*(n+m)), n, m, dt, t[k+1]) for k = 0:N-2]
    X = [view(Z, iX[k]) for k = 1:N]
    U = [view(Z, iU[k]) for k = 1:N-1]
    X_ = view(Z, vcat(iX...))
    U_ = view(Z, vcat(iU...))
    push!(Z_, ViewKnotPoint(X[end], n, m, 0.0, tf))
    Primals(Z,Z_,X,U,X_,U_)
end

@inline Primals(prob::Primals{n,m}) where {n,m} = Primals(n,m, prob.N, prob.tf)
@inline Primals(prob::Problem) = begin Z = Primals(size(prob)..., prob.tf); copyto!(Z, prob.Z); Z; end

function Base.copyto!(Z0::Primals, Z::Traj)
    N = length(Z)
    for k = 1:N-1
        Z0.Z_[k].z .= Z[k].z
    end
    Z0.Z_[N].z .= state(Z[N])
end

@inline Base.copyto!(Z0::Primals, Z::Vector{<:Real}) = copyto!(Z0.Z, Z)
function Base.copy(Z::Primals{n,m}) where {n,m}
    tf = traj(Z)[end].t
    Z_ = Primals(n,m, length(traj(Z)), tf)
    copyto!(Z_, vect(Z))
    return Z_
end

function Base.:+(Z1::Primals, Z2::Primals)
    Z = copy(Z1)
    Z.Z .= Z1.Z .+ Z2.Z
    return Z
end

function Base.:*(a::Real, Z1::Primals)
    Z = copy(Z1)
    Z.Z .= a*Z1.Z
    return Z
end

@inline RobotDynamics.Traj(Z::Primals) = Z.Z_

#---
"""
	TrajData{n,m,T}
"""
struct TrajData{n,m,T}
	xinds::Vector{SVector{n,Int}}
	uinds::Vector{SVector{m,Int}}
	t::Vector{T}
	dt::Vector{T}
end

function TrajData(Z::Traj{n,m}) where {n,m}
	N = length(Z)
	Nu = RobotDynamics.is_terminal(Z[end]) ? N-1 : N
	xinds = [Z[k]._x .+ (k-1)*(n+m) for k = 1:N]
	uinds = [Z[k]._u .+ (k-1)*(n+m) for k  = 1:Nu]
	t = RobotDynamics.get_times(Z)
	dt = [z.dt for z in Z]
	TrajData(xinds, uinds, t, dt)
end

Base.length(Zdata::TrajData) = length(Zdata.xinds)

function RobotDynamics.StaticKnotPoint(Z::Vector, Zdata::TrajData{n,m}, k::Int) where {n,m}
	x = Z[Zdata.xinds[k]]
	if k <= length(Zdata.uinds)
		u = Z[Zdata.uinds[k]]
	else
		u = @SVector zeros(m)
	end
	dt = Zdata.dt[k]
	t = Zdata.t[k]
	StaticKnotPoint(x,u,dt,t)
end

mutable struct NLPTraj{n,m,T} <: RobotDynamics.AbstractTrajectory{n,m,T}
	Z::Vector{T}
	Zdata::TrajData{n,m,Float64}
end

@inline Base.getindex(Z::NLPTraj, k::Int) = StaticKnotPoint(Z.Z, Z.Zdata, k)
@inline Base.iterate(Z::NLPTraj) = length(Z.Zdata) == 0 ? nothing : (Z[1],1)
@inline Base.iterate(Z::NLPTraj, k::Int) = k >= length(Z.Zdata) ? nothing : (Z[k+1],k+1)
@inline Base.length(Z::NLPTraj) = length(Z.Zdata)
@inline Base.size(Z::NLPTraj) = (length(Z.Zdata),)
@inline Base.eltype(Z::NLPTraj{n,m,T}) where {n,m,T} = StaticKnotPoint{n,m,T,n+m}
@inline Base.IteratorSize(Z::NLPTraj) = Base.HasLength()
@inline Base.IteratorEltype(Z::NLPTraj) = Base.HasEltype()
@inline Base.firstindex(Z::NLPTraj) = 1
@inline Base.lastindex(Z::NLPTraj) = length(Z)

#---

"""
	TrajOptNLP{n,m,T}
"""
struct TrajOptNLP{n,m,T}
	model::AbstractModel
	zinds::Vector{UnitRange{Int}}

	# Data
	data::NLPData{T}

	# Objective
	obj::AbstractObjective
	E::Objective{QuadraticViewCost{n,m,T}}

	# Constraints
	conSet::NLPConstraintSet{T}

	# Solution
	Z::NLPTraj{n,m,T}
end

function TrajOptNLP(prob::Problem; remove_bounds::Bool=false)
	n,m,N = size(prob)
	NN = N*n + (N-1)*m  # number of primal variables
	P = sum(num_constraints(prob))

	data = NLPData(NN,P)

	conSet = NLPConstraintSet(prob.model, prob.constraints, data)

	# Remove goal and bound constraints and store them in data.zL and data.zU
	if remove_bounds
		primal_bounds!(data.zL, data.zU, get_constraints(prob), true)
	end

	Zdata = TrajData(prob.Z)
	zinds = [(k-1)*(n+m) .+ (1:n+m) for k = 1:N-1]
	push!(zinds, (N-1)*(n+m) .+ (1:n))

	E = Objective([QuadraticViewCost(
			data.G, data.g, QuadraticCost{Float64}(n, m, terminal=(k==N)),k)
			for k = 1:N])

	Z = Primals(prob)
	dZ = Primals(prob)
	Z̄ = Primals(prob)
	Z = NLPTraj(zeros(NN), Zdata)
	TrajOptNLP(prob.model, zinds, data, prob.obj, E, conSet, Z)
end

num_knotpoints(nlp::TrajOptNLP) = length(nlp.zinds)
@inline num_vars(nlp::TrajOptNLP) = length(nlp.data.g)
@inline num_constraints(nlp::TrajOptNLP) = length(nlp.data.d)

"""
	eval_f(nlp::TrajOptNLP, Z)

Evalate the cost function at `Z`.
"""
function eval_f(nlp::TrajOptNLP, Z=nlp.Z.Z)
	if eltype(Z) !== eltype(nlp.Z.Z)
		Z_ = NLPTraj(Z, nlp.Z.Zdata)
	else
		nlp.Z.Z = 	Z
		Z_ = nlp.Z
	end
	return cost(nlp.obj, Z_)
end

"""
	grad_f!(nlp::TrajOptNLP, Z, g)

Evaluate the gradient of the cost function
"""
function grad_f!(nlp::TrajOptNLP, Z, g=nlp.data.g)
	N = num_knotpoints(nlp)
	nlp.Z.Z = Z
	cost_gradient!(nlp.E, nlp.obj, nlp.Z)
	if g !== nlp.data.g
		g .= nlp.g  # TODO: reset views instead of copying
	end
	return g
end

"""
	hess_f!(nlp::TrajOptNLP, Z, G)

Evaluate the hessian of the cost function `G`.
"""
function hess_f!(nlp::TrajOptNLP, Z, G=nlp.data.G)
	N = num_knotpoints(nlp)
	nlp.Z.Z = Z
	cost_hessian!(nlp.E, nlp.obj, nlp.Z, true)  # TODO: figure out how to not require the reset
	if G !== nlp.data.G
		copyto!(G, nlp.G)  # TODO: reset views instead of copying
		@warn "Copying hessian"
	end
	return G
end

"""
	hess_f_structure(nlp::TrajOptNLP)

Returns a sparse matrix `D` of the same size as the constraint Jacobian, corresponding to
the sparsity pattern of the constraint Jacobian. Additionally, `D[i,j]` is either zero or
a unique index from 1 to `nnz(D)`.
"""
function hess_f_structure(nlp::TrajOptNLP)
	NN = num_vars(nlp)
	N = num_knotpoints(nlp)
	n,m = size(nlp.model)
	G = spzeros(Int, NN, NN)
	if nlp.obj isa Objective{<:DiagonalCostFunction}
		for i = 1:NN
			G[i,i] = i
		end
	else
		zinds = nlp.zinds
		off = 0
		for k = 1:N
			nm = length(zinds[k])
			blk = reshape(1:nm^2, nm, nm)
			view(G, zinds[k], zinds[k]) .= blk .+ off
			off += nm^2
		end
	end
	return G
end

"""
	get_rc(A::SparseMatrixCSC)

Given a matrix `A` specifying the sparsity structure, where each non-zero element of `A`
is a unique integer ranging from 1 to `nnz(A)`, return the list of row-column pairs such that
`A[r[i],c[i]] = i`.
"""
function get_rc(A::SparseMatrixCSC)
    row,col,inds = findnz(A)
    v = sortperm(inds)
    row[v],col[v]
end

"""
	eval_c!(nlp::TrajOptNLP, Z, c)

Evaluate the constraints at `Z`, storing the result in `c`.
"""
function eval_c!(nlp::TrajOptNLP, Z, c=nlp.data.d)
	if eltype(Z) !== eltype(nlp.Z.Z)
		# Back-up if trying to ForwardDiff
		Z_ = NLPTraj(Z, nlp.Z.Zdata)
	else
		nlp.Z.Z = Z
		Z_ = nlp.Z
	end
	evaluate!(nlp.conSet, Z_)
	if c !== nlp.data.d
		copyto!(c, nlp.conSet.d)  # TODO: reset views instead of copying
	end
	return c
end

"""
	jac_c!(nlp::TrajOptNLP, Z, C)

Evaluate the constraint Jacobian at `Z`, storing the result in `C`.
"""
function jac_c!(nlp::TrajOptNLP, Z, C=nlp.data.D)
	nlp.Z.Z = Z
	jacobian!(nlp.conSet, nlp.Z)
	if C !== nlp.data.D
		copyto!(C, nlp.conSet.D)  # TODO: reset views instead of copying
	end
	return C
end

"""
	jac_structure(nlp::TrajOptNLP)

Returns a sparse matrix `D` of the same size as the constraint Jacobian, corresponding to
the sparsity pattern of the constraint Jacobian. Additionally, `D[i,j]` is either zero or
a unique index from 1 to `nnz(D)`.
"""
function jac_structure(nlp::TrajOptNLP)
	N = num_knotpoints(nlp)
	NN = num_vars(nlp)
	P = num_constraints(nlp)
	D = spzeros(Int, P, NN)
	conSet = nlp.conSet
	idx = 0
	for k = 1:N
		for (i,conval) in enumerate(get_convals(conSet))
			p = length(conval.con)
			if k in conval.inds
				for (l,w) in enumerate(widths(conval.con))
					blk_len = p*w
					inds = reshape(idx .+ (1:blk_len), p, w)
					j = _index(conval,k)
					# linds[i][j] = inds
					conval.jac[j,l] .= inds
					idx += blk_len
				end
			end
		end
	end
	copyto!(D, nlp.data.D)
	return D
end

"""
	hess_L(nlp::TrajOptNLP, Z, λ, G)

Calculate the Hessian of the Lagrangian `G`, with the vector of current primal variables `Z`
and dual variables `λ`.
"""
function hess_L!(nlp::TrajOptNLP, Z, λ=nlp.data.λ, G=nlp.data.G)
	nlp.Z.Z = Z
	if λ !== nlp.data.λ
		copyto!(nlp.data.λ)  # TODO: reset views instead of copying
	end

	# Cost hessian
	hess_f!(nlp, Z, G)

	# Add Second-order constraint expansion
	∇jacobian!(nlp.conSet, nlp.Z)

	if G !== nlp.data.G
		copyto!(G, nlp.data.G)  # TODO: reset views instead of copying
	end
	return G
end

"""
	primal_bounds!(nlp::TrajOptNLP, zL, zU)

Get the lower and upper bounds on the primal variables.
"""
function primal_bounds!(nlp::TrajOptNLP, zL=nlp.data.zL, zU=nlp.data.zU)
	if zL !== nlp.data.zL
		zL .= nlp.data.zL
		zU .= nlp.data.zU
		nlp.data.zL = zL
		nlp.data.zU = zU
	end
end

"""
	constraint_type(nlp::TrajOptNLP)

Build a vector of length `IE = num_constraints(nlp)` where `IE[i]` is the type of constraint
for constraint `i`.

Legend:
 - 0 -> Inequality
 - 1 -> Equality
"""
function constraint_type(nlp::TrajOptNLP)
	IE = zeros(Int, num_constraints(nlp))
	constraint_type!(nlp, IE)
end
function constraint_type!(nlp::TrajOptNLP, IE)
	conSet = nlp.conSet
	for i = 1:length(conSet)
		conval = conSet.convals[i]
		cinds = conSet.cinds[i]
		for j = 1:length(cinds)
			v = sense(conval.con) == Equality() ? 1 : 0
			IE[cinds[j]] .= v
		end
	end
	return IE
end
