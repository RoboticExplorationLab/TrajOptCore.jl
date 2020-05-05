using TrajOptCore
using TrajectoryOptimization
using Ipopt
using MathOptInterface
const MOI = MathOptInterface

prob = Problems.DubinsCar(:parallel_park)[1]
TrajOptCore.add_dynamics_constraints!(prob)

nlp = TrajOptCore.TrajOptNLP(prob, remove_bounds=true, jac_type=:vector)
optimizer = Ipopt.Optimizer()
TrajOptCore.solve_MOI(nlp, optimizer)
states(nlp.Z)
optimizer.variable_info
