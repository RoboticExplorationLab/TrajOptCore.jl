
function gen_cartpole_prob()
    model = RobotZoo.Cartpole()
    n,m = size(model)
    N = 101
    tf = 5.
    dt = tf/(N-1)

    Q = 1.0e-2*Diagonal(@SVector ones(n))
    Qf = 100.0*Diagonal(@SVector ones(n))
    R = 1.0e-1*Diagonal(@SVector ones(m))
    x0 = @SVector zeros(n)
    xf = @SVector [0, pi, 0, 0]
    obj = LQRObjective(Q,R,Qf,xf,N)

    u_bnd = 3.0
    bnd = BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
    goal = GoalConstraint(xf)
    conSet = ConstraintList(n,m,N)
    add_constraint!(conSet, bnd, 1:N-1)
    add_constraint!(conSet, goal, N)

    X0 = [@SVector fill(NaN,n) for k = 1:N]
    u0 = @SVector fill(0.01,m)
    U0 = [u0 for k = 1:N-1]
    Z = Traj(X0,U0,dt*ones(N))
    prob = Problem{RK3}(model, obj, conSet, x0, xf, Z, N, 0.0, tf)
    rollout!(prob)
    return prob
end
