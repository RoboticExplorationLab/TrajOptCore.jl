abstract type AbstractExpansion{T} end

struct GradientExpansion{T,N,M} <: AbstractExpansion{T}
	x::SizedVector{N,T,1}
	u::SizedVector{M,T,1}
	function GradientExpansion{T}(n::Int,m::Int) where T
		new{T,n,m}(SizedVector{n}(zeros(T,n)), SizedVector{m}(zeros(T,m)))
	end
end

struct CostExpansion{T,N0,N,M} <: AbstractExpansion{T}
	# Cost Expansion Terms
	x ::SizedVector{N0,T,1}
	xx::SizedMatrix{N0,N0,T,2}
	u ::SizedVector{M,T,1}
	uu::SizedMatrix{M,M,T,2}
	ux::SizedMatrix{M,N0,T,2}

	# Error Expansion Terms
	x_ ::SizedVector{N,T,1}
	xx_::SizedMatrix{N,N,T,2}
	u_ ::SizedVector{M,T,1}
	uu_::SizedMatrix{M,M,T,2}
	ux_::SizedMatrix{M,N,T,2}

	tmp::SizedMatrix{N0,N,T,2}
	x0::SizedVector{N0,T,1}  # gradient of cost function only (no multipliers)
end

function CostExpansion{T}(n0::Int, n::Int, m::Int) where T
	x0  = SizedVector{n0}(zeros(T,n0))
	xx0 = SizedMatrix{n0,n0}(zeros(T,n0,n0))
	u0  = SizedVector{m}(zeros(T,m))
	uu0 = SizedMatrix{m,m}(zeros(T,m,m))
	ux0 = SizedMatrix{m,n0}(zeros(T,m,n0))

	x  = SizedVector{n}(zeros(T,n))
	xx = SizedMatrix{n,n}(zeros(T,n,n))
	u  = SizedVector{m}(zeros(T,m))
	uu = SizedMatrix{m,m}(zeros(T,m,m))
	ux = SizedMatrix{m,n}(zeros(T,m,n))
	tmp = SizedMatrix{n0,n}(zeros(T,n0,n))
	x_ = copy(x0)
	CostExpansion(x0,xx0,u0,uu0,ux0, x, xx, u, uu, ux, tmp, x_)
end

function CostExpansion{T}(n::Int, m::Int) where T
	x  = SizedVector{n}(zeros(T,n))
	xx = SizedMatrix{n,n}(zeros(T,n,n))
	u  = SizedVector{m}(zeros(T,m))
	uu = SizedMatrix{m,m}(zeros(T,m,m))
	ux = SizedMatrix{m,n}(zeros(T,m,n))
	tmp = SizedMatrix{n,n}(zeros(T,n,n))
	x_ = copy(x)
	CostExpansion(x,xx,u,uu,ux, x, xx, u, uu, ux, tmp, x_)
end


struct Expansion{T,N0,N,M} <: AbstractExpansion{T}
	x::SizedVector{N,T,1}
	xx::SizedMatrix{N,N,T,2}
	u::SizedVector{M,T,1}
	uu::SizedMatrix{M,M,T,2}
	ux::SizedMatrix{M,N,T,2}
	tmp::SizedMatrix{N0,N,T,2}
	function Expansion{T}(n::Int) where T
		x = SizedVector{n}(zeros(n))
		xx = SizedMatrix{n,n}(zeros(n,n))
		new{T,n,n,0}(x,xx)
	end
	function Expansion{T}(n::Int,m::Int) where T
		x = SizedVector{n}(zeros(n))
		xx = SizedMatrix{n,n}(zeros(n,n))
		u = SizedVector{m}(zeros(m))
		uu = SizedMatrix{m,m}(zeros(m,m))
		ux = SizedMatrix{m,n}(zeros(m,n))
		new{T,n,n,m}(x,xx,u,uu,ux)
	end
	function Expansion{T}(n0::Int,n::Int,m::Int) where T
		x = SizedVector{n}(zeros(n))
		xx = SizedMatrix{n,n}(zeros(n,n))
		u = SizedVector{m}(zeros(m))
		uu = SizedMatrix{m,m}(zeros(m,m))
		ux = SizedMatrix{m,n}(zeros(m,n))
		tmp = SizedMatrix{n0,n}(zeros(n0,n))
		new{T,n0,n,m}(x,xx,u,uu,ux,tmp)
	end
	function Expansion(
			x::SizedVector{N,T,1},
			xx::SizedMatrix{N,N,T,2},
			u::SizedVector{M,T,1},
			uu::SizedMatrix{M,M,T,2},
			ux::SizedMatrix{M,N,T,2}) where {T,N,M}
		new{T,N,N,M}(x,xx,u,uu,ux)
	end
end
