import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


def encode(outputs, n_out, classes):
    aux = []
    for label in outputs:
        aux.append(classes[label])

    aux = np.array(aux)
    aux = aux.reshape((aux.shape[0], n_out))

    return aux


def load_mnist_dataset(num_examples=None, start=0):
    try:
        print("  - Fetching MNIST Dataset from local cache")
        X = np.loadtxt(fname='./datasets/mnist_images.csv', delimiter=',', max_rows=num_examples)
        y = np.loadtxt(fname='./datasets/mnist_labels.csv', delimiter=',', dtype=object, max_rows=num_examples)
    except Exception:
        print("  - Fetching MNIST Dataset from OpenML")
        # Load data from https://www.openml.org/d/554
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, cache=True)

        np.savetxt(fname='./datasets/mnist_images.csv', X=X, delimiter=',', fmt='%d')
        np.savetxt(fname='./datasets/mnist_labels.csv', X=y, delimiter=',', fmt='%c')

    return X, y


def preprocess_dataset(X, y):
    random_state = check_random_state(0)
    permutation = random_state.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]
    X = X.reshape((X.shape[0], -1))

    return X, y


def split_dataset(X, y, train_size, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def get_accuracy(predictions, labels):
    acc = predictions.T - labels
    acc = np.sum(acc, axis=1, keepdims=True)

    acc = acc[acc == 0]
    acc = acc.shape[0] / labels.shape[0] * 100

    return acc
