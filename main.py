import numpy as np
import pandas
import matplotlib.pyplot as plt

# Gradient Descent for Linear Regression
# yhat = wx + b
# loss = Σ ((y-yhat)^2) / N  i.e mean square error

# Initialise some parameters

# Using random values to train the model
# x = np.random.randn(10,1)
# # y = 2*x + np.random.rand()          # x & y are n-d array which is a numpy obj
# y = 3*x + 5                           # These are actual values we generally use collected data to train


data = pandas.read_csv("dataset.csv")
x = data.iloc[:, 0].to_numpy()          # Convert Pandas series to numpy.ndarray class
y = data.iloc[:, 1].to_numpy()          # even though it's still unnecessary

# Plot the datapoints
fig, axes = plt.subplots()              # Config. a 1 row x 1 column graph
axes.scatter(x, y)  # Draw the points
plt.title("Data Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig("datapoints.png")           # Save the current graph state
fig.delaxes(axes)                       # Discard the current graph to avoid overlapping axes issue

# Parameters
w = 0.0
b = 0.0

# Hyper-parameters
learning_rate = 0.00001


# Create Gradient Descent Function

# Our goal is to minimize the loss/cost function
def descend(x, y, w, b, learning_rate):
    dldw = 0.0  # Derivative of loss wrt to w
    dldb = 0.0  # Derivative of loss wrt to b
    N = x.shape[0]  # Dimension of N=100

    # loss = (y-yhat)^2

    for xi, yi in zip(x, y):
        dldw += -2 * xi * (yi - (w * xi + b))  # Summing all the little error
        dldb += -2 * (yi - (w * xi + b))

    # Updating the value/Adjusting the parameter
    w = w - learning_rate * (1 / N) * dldw
    b = b - learning_rate * (1 / N) * dldb
    return w, b


w_array = []
b_array = []

# Iteratively make updates
for epoch in range(1000):
    w, b = descend(x, y, w, b, learning_rate)
    yhat = w * x + b
    loss = np.divide(np.sum((y - yhat) ** 2, axis=0), x.shape[0])
    print(f"Epoch:{epoch} loss is {loss}, parameters w: {w}, b: {b}")
    w_array.append(w)
    b_array.append(b)

# Plot first graph
plt.subplot(1, 2, 1)                            # Select the 1st graph of 1 row X 2 column graph
plt.title("Best fit for curve")
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(x, y)

# Draw line
slope = w
intercept = b
abline_values = [slope * i + intercept for i in x]
plt.plot(x, abline_values, 'r')

# Plot second graph
wpoints = np.array(w_array)
bpoints = np.array(b_array)
plt.subplot(1, 2, 2)
plt.title("Parameters")
plt.xlabel("Epochs")
plt.plot(bpoints, c="DeepPink", label="b")      # Set color and legend
plt.plot(wpoints, c="Blue", label="w")
plt.legend()                                    # Show legends
plt.grid()                                      # Creates grid to the current graph

# Save graphs
plt.savefig("Graph_transparent.png", transparent=True)
plt.savefig("Graph.png")

# To get full screen graph
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

plt.show()
