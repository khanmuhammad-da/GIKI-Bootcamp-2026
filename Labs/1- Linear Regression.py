import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import math

data = np.loadtxt('ex1data1.txt', delimiter=",")
x_train= data[:,0]
y_train= data[:,1]

print("X_train is ",x_train)
print("y_train is ",y_train)

print("The Shape of X Train is ", x_train.shape)
print("The shape of y Train is ",y_train.shape )

#scatter plot

plt.scatter(x_train,y_train,marker='o',c='r')
# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')

plt.show()

def compute_cost(x,y,w,b):
    """
       Computes the cost function for linear regression.

       Args:
           x (ndarray): Shape (m,) Input to the model (Population of cities)
           y (ndarray): Shape (m,) Label (Actual profits for the cities)
           w, b (scalar): Parameters of the model

       Returns
           total_cost (float): The cost of using w,b as the parameters for linear regression
                  to fit the data points in x and y
       """

    m = x.shape[0]
    total_cost = 0
    cost = 0
    for i in range(m):
        f_wb=w*x[i]+b
        cost += (f_wb-y[i])**2
    total_cost = cost/(2*m)
    return total_cost
def compute_gradient(x,y,w,b):
    """
       Computes the cost function for linear regression.

       Args:
           x (ndarray): Shape (m,) Input to the model (Population of cities)
           y (ndarray): Shape (m,) Label (Actual profits for the cities)
           w, b (scalar): Parameters of the model

       Returns
           total_cost (float): The cost of using w,b as the parameters for linear regression
                  to fit the data points in x and y
       """
    m = x.shape[0]
    dj_dw=0
    dj_db=0

    for i in range(m):
        dj_dw += (w*x[i]+b-y[i])*x[i]
        dj_db += (w*x[i]+b-y[i])
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw,dj_db


# Compute and display gradient with w initialized to zeroes
initial_w =0
initial_b =0

temp_dj_dw, temp_dj_db = compute_gradient(x_train,y_train,initial_w,initial_b)
print("Gradient at initial w,b (zeros): ", temp_dj_dw, temp_dj_db)

# Compute and display cost and gradient with non-zero w

test_w=0.2
test_b=0.2

temp_dj_dw, temp_dj_db = compute_gradient(x_train,y_train,test_w,test_b)
print("Gradient at test w,b (0.2,0.2): ", temp_dj_dw, temp_dj_db)

def gradient_descent(x,y,w_in, b_in, compute_cost, compute_gradient,alph, num_iter):
    """
        Performs batch gradient descent to learn theta. Updates theta by taking
        num_iters gradient steps with learning rate alpha

        Args:
          x :    (ndarray): Shape (m,)
          y :    (ndarray): Shape (m,)
          w_in, b_in : (scalar) Initial values of parameters of the model
          cost_function: function to compute cost
          gradient_function: function to compute the gradient
          alpha : (float) Learning rate
          num_iters : (int) number of iterations to run gradient descent
        Returns
          w : (ndarray): Shape (1,) Updated values of parameters of the model after
              running gradient descent
          b : (scalar)                Updated value of parameter of the model after
              running gradient descent
        """

    #number of training examples
    m = len(x)

    #an array to store cost J and w's at each iteration for graph later
    j_history = []
    w_history = []
    w = copy.deepcopy(w_in)#avoid modifying global w keep original w_in unchnaged
    b = b_in
    for i in range(num_iter):

        #calculate the gradient descent at each iteration
        dj_dw, dj_db = compute_gradient(x,y,w,b)

        #update the parameters using w,b, alph and gradient
        w= w-alph*dj_dw
        b= b-alph*dj_db

        #save cost j at each iteration
        if i<10000: #prevent resource exhaustion
            cost = compute_cost(x,y,w,b)
            j_history.append(cost)

        #print cost at every intervals 10
        if i% math.ceil(num_iter/10)==0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost = {float(j_history[-1]):8.2f}   ")

    return w,b, j_history,w_history

initial_w =0
initial_b =0

iterations = 1500
alpha = 0.01
w,b,j_history,w_history = gradient_descent(x_train,y_train,initial_w,initial_b,compute_cost,compute_gradient,alpha,iterations)
print("w, b found by gradient descent ", w,b)


plt.plot(j_history,color="purple")
plt.xlabel("Iteration")
plt.ylabel("Cost (J)")
plt.title("Convergence of Gradient Descent")
plt.show()

m = x_train.shape[0]
predicted = np.zeros(m)
for i in range(m):
    predicted[i] = w*x_train[i]+b

# Plot the linear fit
plt.plot(x_train, predicted, c = "b")

# Create a scatter plot of the data.
plt.scatter(x_train, y_train, marker='x', c='r')

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()

predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))

predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))







compute_cost(x_train,y_train,1,2)

