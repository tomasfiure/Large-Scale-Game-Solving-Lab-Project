import project as p
import numpy as np
import matplotlib.pyplot as plt

# # Generate random matrix 100 by 10
A = np.random.rand(100, 10)
# Generate random vector with 100 entries
b = np.random.rand(100)

#test for optimal k
#comment out if testing for optimal epsilon
epsilon = 10e-6
eta = 0.001

minimizing_k = 1
curr_mini = 10000
for i in range(1,31):
    k_avg = 0
    for j in range(5):
        print("i:",i,"j:",j)
        # Generate random matrix 100 by 10
        A = np.random.rand(100, 10)
        # Generate random vector with 100 entries
        b = np.random.rand(100)
        
        optimal_solution = np.linalg.lstsq(A, b, rcond=None)[0]
        optimal_objective_value = p.L2sq(A,optimal_solution,b)
        actual_value = p.L2sq(A,p.sgd(A,b,i,p.L2sq,epsilon),b)
        temp_mini = np.abs(optimal_objective_value - actual_value)
        k_avg+=temp_mini
    k_avg*=(1/5)
    print("k_avg:",k_avg,"curr_mini:",curr_mini)
    if (k_avg<curr_mini):
        minimizing_k=i
        curr_mini=k_avg
        
print("optimal k:",minimizing_k)

# #testing did not find an optimal k, so no evidence to suggest there is an optimal k

#test for optimal epsilon
#comment out when testing for optimal k
k=10
eta = 0.001
li = []
for i in range(15):
    minimizing_e = -1
    curr_mini = 10000
    for i in range(1,15):
        k_avg = 0
        for j in range(5):
            print("i:",i,"j:",j)
            # Generate random matrix 100 by 10
            A = np.random.rand(100, 10)
            # Generate random vector with 100 entries
            b = np.random.rand(100)
            
            optimal_solution = np.linalg.lstsq(A, b, rcond=None)[0]
            optimal_objective_value = p.L2sq(A,optimal_solution,b)
            actual_value = p.L2sq(A,p.sgd(A,b,k,p.L2sq,1*10**(-i)),b)
            temp_mini = np.abs(optimal_objective_value - actual_value)
            k_avg+=temp_mini
        k_avg*=(1/10)
        print("k_avg:",k_avg,"curr_mini:",curr_mini)
        if (k_avg<curr_mini):
            minimizing_e=-i
            curr_mini=k_avg
            
    print("optimal e:",minimizing_e)
    li.append(minimizing_e)
print("results:",li)

#testing does not show clear optimal epsilon magnitude
#there is weak vidence showing -10 might be a good epsilon magnitude choice
#notable: smaller epsilon does not necessarily equal more accurate results

#overall testing shows consistent result accurate to at least 10^(-7)