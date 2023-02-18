import numpy as np

def predict(X, w, b):
    return X * w + b

def loss(X, Y, w, b):
    error = predict(X, w, b) - Y
    squared_error = error ** 2
    return np.average(squared_error)

def train(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w, b)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))
        if loss(X, Y, w + lr, b) < current_loss:
            w += lr
        elif loss(X, Y, w - lr, b) < current_loss:
            w -= lr
        elif loss(X, Y, w, b + lr) < current_loss:
            b += lr
        elif loss(X, Y, w, b - lr) < current_loss:
            b -= lr
        else:
            return w, b
    raise Exception("Couldn't converge within %d iterations" % iterations)

# Import Dataset
X, Y = np.loadtxt('pizza.txt', skiprows=1, unpack=True)

# Train the system
w, b = train(X, Y, iterations=1000000, lr=0.0001)
print("\nw=%.3f, b=%.3f" % (w, b))

# Predict
print("Prediction: x=%d => y=%.2f" % (20, predict(20, w, b)))