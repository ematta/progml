import numpy as np

def predict(X, w):
    return np.matmul(X, w)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward(X, w):
    return sigmoid(predict(X, w))


def classify(X, w):
    return np.round(forward(X, w))


def mse_loss(X, Y, w):
    return np.average((forward(X, w) - Y) ** 2)


def loss(X, Y, w):
    y_hat = forward(X, w)
    first_term = Y * np.log(y_hat)
    second_term = (1 - Y) * np.log(1 - y_hat)
    return -np.average(first_term + second_term)


def gradient(X, Y, w):
    return np.matmul(X.T, (forward(X, w) - Y)) / X.shape[0]


def train(X, Y, iterations, lr):
    w = np.zeros((X.shape[1], 1))
    for i in range(iterations):
        print("Iteration %4d => Loss: %.20f" % (i, loss(X, Y, w)))
        w -= gradient(X, Y, w) * lr
    return w


def test(X, Y, w):
    total_examples = X.shape[0]
    correct_results = np.sum(classify(X, w) == Y)
    success_percentage = correct_results / total_examples * 100
    print("Success: %d/%d (%.2f%%)" % (correct_results, total_examples, success_percentage))


# Prepare data
x1, x2, x3, y = np.loadtxt("police.txt", skiprows=1, unpack=True)
X = np.column_stack((np.ones(x1.size), x1, x2, x3))
Y = y.reshape(-1, 1)
w = train(X, Y, iterations=100000, lr=0.001)

# Test
test(X, Y, w)

print("\nWeights: %s" % w.T)
print("\nA few predictions:")
for i in range(5):
    print("X[%d] -> %.4f (label: %d)" % (i, predict(X[i], w), Y[i]))