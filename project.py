import numpy as np

# Generate random matrix 100 by 10
matrix = np.random.rand(100, 10)
# Generate random vector with 100 entries
vector = np.random.rand(100, 1)
# Print the matrix
# print(matrix)
# print(vector)
def L2sq(A,x,b):
    v = np.dot(A,x) + b
    return (np.linalg.norm(v))**2

    
def sgd(A,b,eta,k):
    # Begin Mini-Batch SGD 
    epochs = 1000
    
    # Pick random x to start
    theta_i = np.random.rand(10, 1)
    #mini = L2sq(matrix,x,vector)
    l_sum=0
    A_t = np.transpose(A)
    for i in range(1,epochs):
        for i in range(1,k):
            x = np.random.rand(10, 1)
            l_sum+= 2 * A_t * L2sq(A,x,b)
        l_sum*=1/k
        theta_i = theta_i + eta*l_sum
    
    return L2sq(A,theta_i,b)

print(sgd(matrix,vector,0.01,10)) 
