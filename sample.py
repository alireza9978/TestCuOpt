import cuopt
from cuopt import routing
import cudf

print(f"cuOpt version: {cuopt.__version__}")

# Simple 4-location VRP: depot + 3 customers
n_locations = 4

cost_matrix = cudf.DataFrame([
    [0, 10, 15, 20],
    [10,  0,  35, 25],
    [15, 35,   0, 30],
    [20, 25,  30,  0],
], dtype="float32")

data_model = routing.DataModel(n_locations, n_fleet=1)
data_model.add_cost_matrix(cost_matrix)

solver_settings = routing.SolverSettings()
solver_settings.set_time_limit(5)

solution = routing.Solve(data_model, solver_settings)

if solution.get_status() == 0:
    print("Status: success")
    print(f"Cost:   {solution.get_total_objective():.2f}")
    print(f"Route:\n{solution.get_route()}")
else:
    print(f"Solver status: {solution.get_status()}")
