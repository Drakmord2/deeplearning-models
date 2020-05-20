from models.DeepNeuralNetwork import DeepNeuralNetwork
from utils.util import load_mnist_dataset, preprocess_dataset, split_dataset, encode, get_accuracy

if __name__ == "__main__":
    print("\n\tDeep Neural Network")

    print("\n-Dataset")
    X, y = load_mnist_dataset(num_examples=7000)

    print("  - Preprocessing Dataset")
    X, y = preprocess_dataset(X, y)

    print("  - Splitting Dataset")
    train_size = 6000
    test_size = 1000

    X_train, y_train, X_test, y_test = split_dataset(X, y, train_size, test_size)

    print("  - Encoding labels")
    n_x = X_train.shape[1]
    n_y = 4
    classes = {
            '0': [0, 0, 0, 0],
            '1': [0, 0, 0, 1],
            '2': [0, 0, 1, 0],
            '3': [0, 0, 1, 1],
            '4': [0, 1, 0, 0],
            '5': [0, 1, 0, 1],
            '6': [0, 1, 1, 0],
            '7': [0, 1, 1, 1],
            '8': [1, 0, 0, 0],
            '9': [1, 0, 0, 1]
            }

    y_train = encode(y_train, n_y, classes)
    y_test = encode(y_test, n_y, classes)

    print("\n- Model")
    print("  - Configuring Hyperparameters")
    layers_dims = [n_x, 98, 7, n_y]
    learning_rate = 0.008
    iterations = 3000

    dnn = DeepNeuralNetwork(layers_dims, n_x, n_y, learning_rate, iterations)

    train = True
    if train:
        print("  - Training Model")
        dnn.fit(X_train.T, y_train.T, print_cost=False)
    else:
        print("  - Loading Model Weights from local cache")
        dnn.load_params()

    print("  - Results")
    if train:
        predictions = dnn.predict(X_train.T)
        acc = get_accuracy(predictions, y_train)
        print("    - Training Accuracy: " + str(acc) + "%")

    predictions = dnn.predict(X_test.T)
    acc = get_accuracy(predictions, y_test)
    print("    - Test Accuracy: "+str(acc)+"%")
