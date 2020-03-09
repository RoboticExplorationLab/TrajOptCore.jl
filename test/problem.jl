
# model
model = RobotZoo.Acrobot()
n,m = size(model)

# discretization
tf = 5.0
N = 101

# initial and final conditions
x0 = @SVector [-pi/2, 0, 0, 0]
xf = @SVector [+pi/2, 0, 0, 0]

# objective
Q = Diagonal(@SVector fill(1.0, n))
R = Diagonal(@SVector fill(0.01, m))
Qf = 100*Q
obj = LQRObjective(Q,R,Qf,xf,N)

# constraints
conSet = ConstraintSet(n,m,N)
goal = GoalConstraint(xf)
bnd  = BoundConstraint(n,m, u_min=-15, u_max=15)
add_constraint!(conSet, goal, N:N)
add_constraint!(conSet, bnd, 1:N-1)

# initialization
u0 = @SVector fill(0.0,m)

# set up problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
@test prob.x0 == x0
@test prob.xf == xf
@test integration(prob) == Dynamics.RK3
@test controls(prob) == [u0 for k = 1:N-1]
prob2 = change_integration(prob, Dynamics.RK2)
@test integration(prob2) == Dynamics.RK2
@test prob.x0 == x0
@test prob.xf == xf

rollout!(prob)
@test states(prob)[1] == x0
@test isfinite(cost(prob))
