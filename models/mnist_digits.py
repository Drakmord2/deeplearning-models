from algorithms.DeepNeuralNetwork import DeepNeuralNetwork
from algorithms.TensorflowDNN import TensorflowDNN
from algorithms.KerasCNN import KerasCNN
from utils.util import load_mnist_dataset, preprocess_dataset, split_dataset, encode


class MNIST:
    def __init__(self, num_examples, train_size, model_type):
        print("\n-Dataset")
        X, Y = load_mnist_dataset(num_examples=num_examples)

        print("  - Preprocessing Dataset")
        X, Y = preprocess_dataset(X, Y)

        print("  - Splitting Dataset")
        test_size = num_examples - train_size
        assert test_size > 0

        self.X_train, self.Y_train, self.X_test, self.Y_test = split_dataset(X, Y, train_size, test_size)
        self.num_inputs = self.X_train.shape[1]
        self.num_outputs = 10
        self.encode_labels()

        if model_type == 'tf-cnn':
            self.X_train = self.X_train.reshape((self.X_train.shape[0], 28, 28, 1))
            self.X_test = self.X_test.reshape((self.X_test.shape[0], 28, 28, 1))
            self.num_inputs = self.X_train.shape[1:]

    def encode_labels(self):
        print("  - Encoding labels")
        classes = {
            '0': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            '1': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            '2': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            '3': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            '4': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            '5': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            '6': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            '7': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            '8': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            '9': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }

        self.Y_train = encode(self.Y_train, self.num_outputs, classes)
        self.Y_test = encode(self.Y_test, self.num_outputs, classes)

    def get_model(self, model):

        if model == "dnn":
            print("\n\tDeep Neural Network")

            print("\n- Model")
            print("  - Configuring Hyperparameters")
            layers_dims = [self.num_inputs, 800, self.num_outputs]
            learning_rate = 0.00002
            iterations = 150
            print("    - Layers: {} | Learning Rate: {} | Iterations: {} | Examples: {}".format(layers_dims, learning_rate, iterations, self.X_train.shape[0]))

            dnn = DeepNeuralNetwork(layers_dims, self.num_inputs, self.num_outputs, learning_rate, iterations)

            return dnn

        if model == "tf-dnn":
            print("\n\tTensorFlow Deep Neural Network")

            print("\n- Model")
            print("  - Configuring Hyperparameters")
            layers_dims = [self.num_inputs, 800, self.num_outputs]
            learning_rate = 0.00002
            iterations = 150
            print("    - Layers: {} | Learning Rate: {} | Iterations: {} | Examples: {}".format(layers_dims, learning_rate, iterations, self.X_train.shape[0]))

            tfdnn = TensorflowDNN(layers_dims, self.num_inputs, self.num_outputs, learning_rate, iterations)

            return tfdnn

        if model == "tf-cnn":
            print("\n\tTensorFlow Convolutional Neural Network")

            print("\n- Model")
            print("  - Configuring Hyperparameters")
            iterations = 5
            print("    - Iterations: {} | Examples: {}".format(iterations, self.X_train.shape[0]))

            tfcnn = KerasCNN(self.num_inputs, iterations)

            return tfcnn

        raise NotImplementedError()
