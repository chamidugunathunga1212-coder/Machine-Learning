import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate        # learning rate
        self.n_iters = n_iters         # number of training iterations
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # initialize weights and bias with zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # training
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # linear output
                linear_output = np.dot(x_i, self.weights) + self.bias
                # apply activation function
                y_predicted = self.activation_func(linear_output)

                # perceptron update rule
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)


# Example usage
if __name__ == "__main__":
    # Simple dataset: OR logic gate
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 1, 1, 1])  # OR gate output

    perceptron = Perceptron(learning_rate=0.1, n_iters=10)
    perceptron.fit(X, y)

    predictions = perceptron.predict(X)
    print("Predictions:", predictions)
