# Simple scatter plot of the following two objects
# X: [6.09 5.56 5.85 5.69 5.01]
# y: [7.62 4.88 6.32 9.28 4.96]

import matplotlib.pyplot as plt
import numpy as np

X = np.array([16.09, 15.56, 15.85, 15.69, 15.01])
y = np.array([17.62, 14.88, 16.32, 16.28, 14.96])

plt.scatter(X, y)
plt.xlabel('X: Temperature yesterday')
plt.ylabel('Y: Temperature today')
plt.title('Linear regression')
plt.show()

# Compute the residual sum of squares for a simple linear regression model
b = 0
w = 0.5
sum_of_squares = np.sum((y - (b + w * X)) ** 2)
print(sum_of_squares)
# Residual sum of squares for b=0, w=0.5: 339.3

b = 1
w = 1
sum_of_squares = np.sum((y - (b + w * X)) ** 2)
print(sum_of_squares)
# Residual sum of squares for b=1, w=1.: 4.6

# Define function for computing gradients
def compute_gradients(X, y, w, b):
    N = len(y)
    y_pred = w * X + b
    dw = (-2/N) * np.sum(X * (y - y_pred))
    db = (-2/N) * np.sum(y - y_pred)
    return dw, db

# Define function for rmse
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Define linear regression prediction function
def predict(X, w, b):
    return w * X + b

w, b = 0.0, 0.0
alpha = 0.001
epochs = 100
losses = []

for epoch in range(epochs):
    dw, db = compute_gradients(X, y, w, b)
    w -= alpha * dw
    b -= alpha * db
    loss = rmse(y, predict(X, w, b))
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}, w: {w:.4f}, b: {b:.4f}')
    losses.append(loss)

# Plot the fitted values as a line on top of the data
plt.scatter(X, y, label='Data points')
plt.plot(X, predict(X, w, b), color='red', label='Fitted line')
plt.xlabel('X: Temperature yesterday')
plt.ylabel('Y: Temperature today')
plt.title('Linear regression')
plt.legend()
plt.show()

# Re-run the code but this time store the weights and biases
w, b = 0.0, 0.0
alpha = 0.001
epochs = 100
losses = []
weights = []
biases = []

for epoch in range(epochs):
    dw, db = compute_gradients(X, y, w, b)
    w -= alpha * dw
    b -= alpha * db
    loss = rmse(y, predict(X, w, b))
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}, w: {w:.4f}, b: {b:.4f}')
    losses.append(loss)
    weights.append(w)
    biases.append(b)

# Plot the parameter trajectories
plt.plot(weights, biases, marker='o')
plt.xlabel('Weight (w)')
plt.ylabel('Bias (b)')
plt.title('Parameter Trajectory')
plt.grid()
plt.show()

# Plot the whole loss surface
W, B = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
Z = np.array([rmse(y, predict(X, w, b)) for w, b in zip(np.ravel(W), np.ravel(B))])
Z = Z.reshape(W.shape)

plt.contourf(W, B, Z, levels=50, cmap='viridis')
plt.colorbar(label='RMSE')
plt.scatter(weights, biases, color='red', marker='o')
plt.xlabel('Weight (w)')
plt.ylabel('Bias (b)')
plt.title('Loss Surface')
plt.show()


##################################################

# Now extend to having two X values

# First get a new X variable - representing rainfall
X2 = np.array([1.4, 4.6, 0.0, 0.0, 2.1])

# Compute example residual sum of squares for a simple linear regression model
b = 0
w1 = 0.5
w2 = 0.3
sum_of_squares = np.sum((y - (b + w1 * X + w2 * X2)) ** 2)
print(sum_of_squares)

# Define linear regression prediction function
def predict(X, w1, w2, b):
    return w1 * X + w2 * X2 + b

# Need new compute_gradients function
def compute_gradients(X, y, w1, w2, b):
    N = len(y)
    y_pred = predict(X, w1, w2, b)
    dw1 = (-2/N) * np.sum(X * (y - y_pred))
    dw2 = (-2/N) * np.sum(X2 * (y - y_pred))
    db = (-2/N) * np.sum(y - y_pred)
    return dw1, dw2, db

w1, w2, b = 0.0, 0.0, 0.0
alpha = 0.001
epochs = 100
losses = []

# Go through epochs
for epoch in range(epochs):
    dw1, dw2, db = compute_gradients(X, y, w1, w2, b)
    w1 -= alpha * dw1
    w2 -= alpha * dw2
    b -= alpha * db
    loss = rmse(y, predict(X, w1, w2, b))
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}, w1: {w1:.4f}, w2: {w2:.4f}, b: {b:.4f}')
    losses.append(loss)

# Plot the fitted values as a line on top of the data
plt.scatter(X, y, label='Data points')
plt.scatter(X, predict(X, w1, w2, b), color='red', label='Fitted points')
plt.xlabel('X: Temperature yesterday')
plt.ylabel('Y: Temperature today')
plt.title('Linear regression')
plt.legend()
plt.show()

##################################################

# Next example: y is defined as either hot (1) or cold (0)
y = np.array([1, 0, 1, 1, 0])

# Use a binary cross entropy loss function
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Compute example residual sum of squares for a simple linear regression model
b = 0
w1 = 0.5
w2 = 0.3

# Define an activation function we can use for prediction
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Show an example of using the cross entropy function
y_pred = sigmoid(w1 * X + w2 * X2 + b)
loss = binary_cross_entropy(y, y_pred)
print(f'Initial loss: {loss:.4f}')

# New gradient function for binary cross entropy lss
def compute_gradients(X, y, w1, w2, b):
    N = len(y)
    y_pred = sigmoid(w1 * X + w2 * X2 + b)
    dw1 = (-2/N) * np.sum(X * (y - y_pred))
    dw2 = (-2/N) * np.sum(X2 * (y - y_pred))
    db = (-2/N) * np.sum(y - y_pred)
    return dw1, dw2, db

# Now create starting values
w1, w2, b = 0.0, 0.0, 0.0
alpha = 0.01
epochs = 100
losses = []

# And start epochs to optimise parameters
for epoch in range(epochs):
    y_pred = sigmoid(w1 * X + w2 * X2 + b)
    loss = binary_cross_entropy(y, y_pred)
    w1 -= alpha * dw1
    w2 -= alpha * dw2
    b -= alpha * db
    loss = binary_cross_entropy(y, y_pred)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}, w1: {w1:.4f}, w2: {w2:.4f}, b: {b:.4f}')
    losses.append(loss)

# Finally plot the predictions
plt.scatter(X, y, label='Data points')
plt.scatter(X, y_pred, color='red', label='Fitted points')
plt.xlabel('X: Temperature yesterday')
plt.ylabel('Y: Temperature today')
plt.title('Logistic regression')
plt.legend()
plt.show()
