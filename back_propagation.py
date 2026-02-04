import numpy as np

np.random.seed(42)


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


input_size = 2
hidden_size = 8
num_hidden_layers = 7
output_size = 1
learning_rate = 0.01
epochs = 700


layer_sizes = (
    [input_size] +
    [hidden_size] * num_hidden_layers +
    [output_size]
)

weights = []
biases = []

for i in range(len(layer_sizes) - 1):
    weights.append(
        np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.1
    )
    biases.append(
        np.zeros((layer_sizes[i+1], 1))
    )


X = np.array([[1.0], [2.0]])
y = np.array([[3.0]])


for epoch in range(epochs):


    activations = [X]
    zs = []

    for i in range(len(weights)):
        z = weights[i] @ activations[i] + biases[i]
        zs.append(z)

        if i == len(weights) - 1:
            a = z             
        else:
            a = relu(z)

        activations.append(a)

    y_pred = activations[-1]

    loss = mse(y, y_pred)

    deltas = [None] * len(weights)


    deltas[-1] = mse_derivative(y, y_pred)


    for i in reversed(range(len(weights) - 1)):
        deltas[i] = (
            weights[i + 1].T @ deltas[i + 1]
        ) * relu_derivative(zs[i])


    for i in range(len(weights)):
        dW = deltas[i] @ activations[i].T
        db = deltas[i]

        weights[i] -= learning_rate * dW
        biases[i] -= learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

print("\nFinal prediction:", y_pred)
