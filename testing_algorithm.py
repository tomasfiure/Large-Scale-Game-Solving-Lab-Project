import numpy as np
import project as p

results = np.zeros(20)
for i in range(20):
    # Generate random matrix 100 by 10
    A = np.random.rand(100, 10)
    # Generate random vector with 100 entries
    b = np.random.rand(100)
    objective_value = p.L2sq(A,p.sgd(A,b,10,p.L2sq),b)
    # Compute optimal solution using least squares
    optimal_solution = np.linalg.lstsq(A, b, rcond=None)[0]
    optimal_objective_value = p.L2sq(A,optimal_solution,b)
    
    # Print results
    print("Objective value from SGD:", objective_value)
    print("Objective value from optimal solution:", optimal_objective_value)
    
    results[i] = np.abs(objective_value - optimal_objective_value)

print("mean error:", np.mean(results))
print("standard deviation of error:", np.std(results))

#mean error seems to be around 1*10^-8