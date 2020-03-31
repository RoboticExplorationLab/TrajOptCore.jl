############################################################################################
#                              OBJECTIVES                                                  #
############################################################################################

abstract type AbstractObjective end
Base.length(obj::AbstractObjective) = length(obj.cost)
state_dim(obj::AbstractObjective) = throw(ErrorException("state_dim not implemented"))
control_dim(obj::AbstractObjective) = throw(ErrorException("control_dim not implemented"))
get_J(obj::AbstractObjective) = throw(ErrorException("get_J not implemented"))

"""```
cost(obj::Objective, Z::Traj)::Float64
cost(obj::Objective, dyn_con::DynamicsConstraint{Q}, Z::Traj)
```
Evaluate the cost for a trajectory.
Calculate the cost gradient for an entire trajectory. If a dynamics constraint is given,
    use the appropriate integration rule, if defined.
"""
function cost(obj::AbstractObjective, Z)
    cost!(obj, Z)
    J = get_J(obj)
    return sum(J)
end


"""$(TYPEDEF)
Objective: stores stage cost(s) and terminal cost functions

Constructors:
```julia
Objective(cost, N)
Objective(cost, cost_term, N)
Objective(costs::Vector{<:CostFunction}, cost_term)
Objective(costs::Vector{<:CostFunction})
```
"""
struct Objective{C} <: AbstractObjective
    cost::Vector{C}
    J::Vector{Float64}
    const_grad::Vector{Bool}
    const_hess::Vector{Bool}
    function Objective(cost::Vector{C}) where C <: CostFunction
        N = length(cost)
        J = zeros(N)
        grad = zeros(Bool,N)
        hess = zeros(Bool,N)
        new{C}(cost, J, grad, hess)
    end
end

state_dim(obj::Objective) = state_dim(obj.cost[1])
control_dim(obj::Objective) = control_dim(obj.cost[1])

# Constructors
function Objective(cost::CostFunction,N::Int)
    Objective([cost for k = 1:N])
end

function Objective(cost::CostFunction,cost_terminal::CostFunction,N::Int)
    Objective([k < N ? cost : cost_terminal for k = 1:N])
end

function Objective(cost::Vector{<:CostFunction},cost_terminal::CostFunction)
    N = length(cost) + 1
    Objective([cost...,cost_terminal])
end

# Methods
"Get the vector of costs at each knot point. `sum(get_J(obj))` is equal to the cost"
get_J(obj::Objective) = obj.J

Base.copy(obj::Objective) = Objective(copy(obj.cost))

Base.getindex(obj::Objective,i::Int) = obj.cost[i]

@inline Base.firstindex(obj::Objective) = firstindex(obj.cost)
@inline Base.lastindex(obj::Objective) = lastindex(obj.cost)
@inline Base.iterate(obj::Objective, start=1) = Base.iterate(obj.cost, start)
@inline Base.eltype(obj::Objective) = eltype(obj.cost)
@inline Base.length(obj::Objective) = length(obj.cost)
Base.IteratorSize(obj::Objective) = Base.HasLength()

Base.show(io::IO, obj::Objective{C}) where C = print(io,"Objective")

const QuadraticObjective{n,m,T} = Objective{QuadraticCost{n,m,T,SizedMatrix{n,n,T,2},SizedMatrix{m,m,T,2}}}

QuadraticObjective(obj::Objective) = Objective(QuadraticCostFunction.(obj.cost))
function QuadraticObjective(obj::AbstractObjective)
    N = length(get_J(obj))
    n = state_dim(obj)
    m = control_dim(obj)
    Objective([QuadraticCost(n,m) for k = 1:N])
end

function QuadraticObjective(n::Int, m::Int, N::Int)
    costfun = QuadraticCost{Float64}(n,m)
    Objective([copy(costfun) for k = 1:N])
end

function QuadraticObjective(obj::QuadraticObjective, model::AbstractModel)
    # Create QuadraticObjective linked to error cost expansion
    @assert RobotDynamics.state_diff_size(model) == size(model)[1]
    return obj
end

function QuadraticObjective(obj::QuadraticObjective, model::LieGroupModel)
    # Create QuadraticObjective partially linked to error cost expansion
    @assert length(obj[1].q) == RobotDynamics.state_diff_size(model)
    n,m = size(model)
    costfuns = map(obj.cost) do costfun
        Q = SizedMatrix{n,n}(zeros(n,n))
        R = costfun.R
        H = SizedMatrix{m,n}(zeros(m,n))
        q = @MVector zeros(n)
        r = costfun.r
        c = costfun.c
        QuadraticCost(Q,R,H,q,r,c, checks=false, terminal=costfun.terminal)
    end
    Objective(costfuns)
end

# Convenience constructors
@doc raw"""```julia
LQRObjective(Q, R, Qf, xf, N)
```
Create an objective of the form
`` (x_N - x_f)^T Q_f (x_N - x_f) + \sum_{k=0}^{N-1} (x_k-x_f)^T Q (x_k-x_f) + u_k^T R u_k``
"""
function LQRObjective(Q::AbstractArray, R::AbstractArray, Qf::AbstractArray,
        xf::AbstractVector, N::Int; checks=true, uf=zeros(size(R,1)))
    H = zeros(size(R,1),size(Q,1))
    q = -Q*xf
    r = -R*uf
    c = 0.5*xf'*Q*xf + 0.5*uf'R*uf
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf

    ℓ = QuadraticCost(Q, R, H, q, r, c, checks=checks)
    ℓN = QuadraticCost(Qf, qf, cf, check=checks, terminal=true)

    Objective(ℓ, ℓN, N)
end

function LQRObjective(
        Q::Union{Diagonal{T,<:SVector{n}},SMatrix{n,n}},
        R::Union{Diagonal{T,<:SVector{m}},SMatrix{m,m}},
        Qf::AbstractArray, xf::AbstractVector, N::Int;
        uf=(@SVector zeros(m)),
        checks=true) where {T,n,m}
    q = -Q*xf
    r = -R*uf
    c = 0.5*xf'*Q*xf + 0.5*uf'R*uf
    qf = -Qf*xf
    cf = 0.5*xf'*Qf*xf

    ℓ = DiagonalCost(Q, R, q, r, c, false)

    ℓN = DiagonalCost(Qf, R, qf, r, cf, true)

    Objective(ℓ, ℓN, N)
end
