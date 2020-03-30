
# struct Jacobian{n0,n,T,S1,S2}
#     ∇f::S1           # "raw" Jacobian, not accounting for G
#     F::S2            # "correct" error Jacobian
#     G::SizedMatrix{n0,n,T,2}
#     rmul::Bool
#     Gt::Transpose{T,SizedMatrix{n0,n,T,2}}
#     lmul::Bool
#     tmp::SizedMatrix{n0,n,T,2}
#     function Jacobian(∇f::J, F::E, G::SizedMatrix{n0,n,T,2}, rmul::Bool) where {n0,n,T,J,E}
#         if rmul && (size(∇f,2) != n0 || size(∇f,1) != size(F,1) || size(F,2) != n)
#             throw(DimensionMismatch("$(size(∇f)) × $(size(G)) → $(size(F))"))
#         end
#         new{n0,n,T,J,E}(∇f, F, G, rmul)
#     end
#     function Jacobian(∇f::J, F::E, G::SizedMatrix{n0,n,T,2}, rmul::Bool,
#             Gt::Transpose, lmul::Bool) where {n0,n,T,J,E}
#         if rmul && (size(∇f,2) != n0 || size(F,2) != n)
#             throw(DimensionMismatch("$(size(Gt)) × $(size(∇f)) × $(size(G)) → $(size(F))"))
#         elseif lmul && (size(∇f,1) != n0 || size(F,1) != n)
#             throw(DimensionMismatch("$(size(Gt)) × $(size(∇f)) × $(size(G)) → $(size(F))"))
#         end
#         if rmul && lmul
#             tmp = zero(G)
#             new{n0,n,T,J,E}(∇f, F, G, rmul, Gt, lmul, tmp)
#         else
#             new{n0,n,T,J,E}(∇f, F, G, rmul, Gt, lmul)
#         end
#     end
# end
#
# function Jacobian(F, G, rmul::Bool, G2, lmul::Bool)
#     n,n̄ = size(G)
#     if rmul && lmul
#         @assert size(F) == (n̄,n̄)
#         ∇f = SizedMatrix{n0,n0}(zeros(n0,n0))
#     elseif rmul
#         p = size(F,1)
#         ∇f = SizedMatrix{p,n0}(zeros(p,n0))
#     elseif lmul
#         p = size(F,2)
#         if p == 1
#             ∇f = SizedVector{n0}(zeros(n0))
#         else
#             ∇f = SizedMatrix{n0,p}(zeros(n0,p))
#         end
#     else
#         @assert n0 == n "Cannot skip error expansion if n0 and n are different sizes"
#         ∇f = F
#     end
#     Jacobian(∇f, F, G, rmul, Transpose(G2), lmul)
# end
#
# function Jacobian(F, G, rmul::Bool = G isa UniformScaling)
#     n0,n = size(G)
#     if rmul
#         p = size(F,1)
#         ∇f = SizedMatrix{p,n0}(zeros(p,n0))
#     else
#         @assert n0 == n
#         ∇f = F
#     end
#     Jacobian(∇f, F, G, rmul)
# end
#
# function error_expansion!(jac::Jacobian)
#     if !jac.lmul && !jac.rmul
#         return nothing
#     elseif jac.lmul && jac.rmul
#         # F = G'∇f*G
#         mul!(jac.tmp, jac.∇f, jac.G)
#         mul!(jac.F, jac.Gt, jac.tmp)
#     elseif jac.rmul
#         # F = ∇f*G
#         mul!(jac.F, jac.∇f, jac.G)
#     elseif jac.lmul
#         # F = G'∇f
#         mul!(jac.F, jac.Gt, jac.∇f)
#     end
#     return nothing
# end
#
# function build_constraint_jacobians(con::AbstractConstraint, inds, G, useG::Bool)
#     n0,n = size(G)
#     p = length(con)
#     map(inds) do k
#         C = SizedMatrix{p,n}(zeros(p,n))
#         Jacobian(C, G[k], useG)
#     end
# end
#
# function build_constraint_jacobians(con::DynamicsConstraint, inds, G, useG::Bool)
#     n0,n = size(G)
#     map(inds) do k
#         C = SizedMatrix{n,n}(zeros(n,n))
#         Jacobian(C, G[k], useG, G[k+1], useG)
#     end
# end
#
# function build_constraint_jacobian_views(D, cinds, zinds, con::AbstractConstraint, inds, G, useG)
#     n0,n = size(G)
#     p = length(con)
#     NN = size(D,2)
#     N = length(G)
#     map(enumerate(inds)) do (i,k)
#         C = view(D,cinds[i],zinds[k])
#         Jacobian(C, G[k], useG)
#     end
# end

@inline get_data(A::AbstractArray) = A
@inline get_data(A::SizedArray) = A.data

struct ConVal{C,V,M,W}
    con::C
    inds::UnitRange{Int}
    vals::Vector{V}
    jac::Matrix{M}
    ∇x::Matrix{W}
    ∇u::Matrix{W}
    c_max::Vector{Float64}
	iserr::Bool  # are the Jacobians on the error state
    function ConVal(n::Int, m::Int, con::AbstractConstraint, inds::UnitRange, jac, vals, iserr::Bool=false)
		if !iserr && size(gen_jacobian(con)) != size(jac[1])
			throw(DimensionMismatch("size of jac[i] does not match the expected size of $(size(gen_jacobian(con)))"))
		end
        p = length(con)
        P = length(vals)
        ix = 1:n
        iu = n .+ (1:m)
		views = [TrajOptCore.gen_views(∇c, con, n, m) for ∇c in jac]
		∇x = [v[1] for v in views]
		∇u = [v[2] for v in views]
        c_max = zeros(P)
        new{typeof(con), eltype(vals), eltype(jac), eltype(∇x)}(con,
			inds, vals, jac, ∇x, ∇u, c_max, iserr)
    end
end

function ConVal(n::Int, m::Int, cval::ConVal)
	# create a ConVal for the "raw" Jacobians, if needed
	# 	otherwise return the same ConVal
	if cval.iserr
		p = length(cval.con)
		ws = widths(cval.con, n, m)
		jac = [SizedMatrix{p,w}(zeros(p,w)) for k in cval.inds, w in ws]
		ConVal(n, m, cval.con, cval.inds, jac, cval.vals, false)
	else
		return cval
	end
end

function _index(cval::ConVal, k::Int)
	if k ∈ cval.inds
		return k - con.inds[1] + 1
	else
		return 0
	end
end

function evaluate!(cval::ConVal, Z::Traj)
    for (i,k) in enumerate(cval.inds)
        cval.vals[i] .= evaluate(cval.con, Z[k])
    end
end

function jacobian!(cval::ConVal, Z::Traj)
	if cval.iserr
		throw(ErrorException("Can't evaluate Jacobians directly on the error state Jacobians"))
	else
	    for (i,k) in enumerate(cval.inds)
	        jacobian!(cval.jac[i], cval.con, Z[k])
	    end
	end
end

@inline violation(::Equality, v) = norm(v,Inf)
@inline violation(::Inequality, v) = maximum(v)

function max_violation(cval::ConVal)
	max_violation!(cval)
    return maximum(cval.c_max)
end

function max_violation!(cval::ConVal)
	s = sense(cval.con)
    for i in eachindex(cval.inds)
        cval.c_max[i] = violation(s, cval.vals[i])
    end
end

function error_expansion!(errval::ConVal, conval::ConVal, model::AbstractModel, G)
	if errval.jac !== conval.jac
		for (i,k) in enumerate(conval.inds)
			mul!(errval.∇x[i], conval.∇x[i], get_data(G[k]))
			errval.∇u[i] .= conval.∇u[i]
		end
	end
end

function error_expansion!(con::AbstractConstraint, err, jac, G)
	mul!(err, jac, G)
end

function gen_convals(n̄::Int, m::Int, con::AbstractConstraint, inds)
    # n is the state diff size
    p = length(con)
	ws = widths(con, n̄,m)
    C = [SizedMatrix{p,w}(zeros(p,w)) for k in inds, w in ws]
    c = [@MVector zeros(p) for k in inds]
    return C, c
end

function gen_convals(D::AbstractMatrix, d::AbstractVector, cinds, zinds, con::AbstractConstraint, inds)
    P = length(inds)
    p = length(con)
    ws = widths(contype(con), n, m)
    C = [view(D, cinds[i], zinds[k,j]) for i in 1:P, j = 1:length(ws)]
    c = [view(d, cinds[i]) for i in 1:P]
    return C,c
end

function gen_convals(blocks::Vector, Yinds, yinds, con::AbstractConstraint, inds)
    C = map(enumerate(inds)) do (i,k)
        nm = size(blocks[k].Y)
        view(blocks[k], cinds[i], 1:nm)
    end
    c = [view(blocks[k].y, yinds[i]) for (i,k) in enumerate(inds)]
    return C,c
end
