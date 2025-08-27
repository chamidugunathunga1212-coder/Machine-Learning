import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # convert labels {0,1} (binary classification)
        y_ = np.where(y <= 0, 0, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # update rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)


# ----------------------
# Run Perceptron on Iris
# ----------------------
if __name__ == "__main__":
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # For perceptron, we do binary classification
    # Let's classify: "Setosa (0)" vs "Not-Setosa (1)"
    y_binary = np.where(y == 0, 0, 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )

    # Scale features (important for perceptron)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train perceptron
    perceptron = Perceptron(learning_rate=0.01, n_iters=1000)
    perceptron.fit(X_train, y_train)

    # Predictions
    predictions = perceptron.predict(X_test)

    # Accuracy
    print("Predictions:", predictions)
    print("True labels:", y_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
