import numpy as np

# Generate random matrix 100 by 10
A = np.random.rand(100, 10)
# Generate random vector with 100 entries
b = np.random.rand(100)

def L2sq(mat,x,vec):
    v = np.dot(mat,x) - vec
    return (np.linalg.norm(v))**2


#implement numerical method to calulate gradient at a point for any loss function
def calculate_gradient(f, mat, x, vec):
    # Calculate the gradient
    epsilon = 1e-5  # Small value for numerical differentiation
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        grad[i] = (f(mat,x_plus,vec) - f(mat,x_minus,vec)) / (2 * epsilon)

    return grad

def sgd(A,b,eta,k,loss):
    # Begin Mini-Batch SGD 
    epochs = 1000
    
    # Pick random x to start
    theta = np.random.rand(10)
    #print("theta:",theta)
    #iterate on epochs (updates to theta)
    for epoch in range(epochs):
        
        #pick indices for minibatch: reorder indices 0 to 99 in random
        #reorder them in the same array where each is there only once to asssure one pass through every index and no double counting
        random_indices = np.arange(100)
        np.random.shuffle(random_indices)
        
        #initialize sum variable to hold gradient sums
        l_sum=np.zeros(10)
        
        #iterate over random arrangement of indices
        #each iteration will take k number of points, so i is increased by k for right index within iteration
        for i in range(0,100,k):
            #print("i:",i,"i+k:",i+k)
            #take sample of data points, randomly assgined by random assignment of indices
            A_k = A[random_indices[i:i+k]]
            b_k = b[random_indices[i:i+k]]
            
            #compute gradient with current sample points and current theta
            #implement numerical method to get gradient on input loss function
            loss_grad = calculate_gradient(loss,A_k,theta,b_k)
            
            #add to sum (each of the ten spots in the vector is its own sum)
            l_sum+=loss_grad
        
        #update theta
        #note 1
        theta-=eta*l_sum    
          
    return theta

# lines of code to check accuracy taken from chat GPT
# Compute L2^2 value obtained by algorithm
# note 2
objective_value = L2sq(A,sgd(A,b,0.001,10,L2sq),b)
# Compute optimal solution using least squares
optimal_solution = np.linalg.lstsq(A, b, rcond=None)[0]
optimal_objective_value = L2sq(A,optimal_solution,b)

# Print results
print("Objective value from SGD:", objective_value)
print("Objective value from optimal solution:", optimal_objective_value)

#potentially graph results over large sample size of trials

#note 1: from what I understand from prof. Kroer's notes, there should be a (1/k) term, but algorithm works very well without it
#note 2: here we are asked to minimize L2^2, so the loss function is f^2. However, algorithm is written for any loss function, just replace input

