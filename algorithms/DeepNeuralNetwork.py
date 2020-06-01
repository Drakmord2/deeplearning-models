#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(27)


class DeepNeuralNetwork:
    def __init__(self, layers_dims, n_x, n_y, learning_rate=0.0075, num_iterations=3000, beta1=0.9, beta2=0.999):
        self.n_x = n_x                          # Number of features
        self.n_y = n_y                          # Number of output units
        self.layers_dims = layers_dims          # Number of units in each layer
        self.learning_rate = learning_rate      # Learning rate for weight update
        self.num_iterations = num_iterations
        self.beta1 = beta1
        self.beta2 = beta2
        self.parameters = []
        self.costs = []

    def fit(self, X, Y, optimizer="adam", print_cost=False):
        # Parameters initialization.
        parameters = self.initialize_parameters(self.layers_dims)

        if optimizer == "adam":
            parameters, costs = self.optimized_training(X, Y, parameters, print_cost)
        else:
            parameters, costs = self.training(X, Y, parameters, print_cost)

        np.save("./outputs/dnn-params.npy", parameters)
        self.parameters = parameters
        self.costs = costs

    def predict(self, X):
        m = X.shape[1]
        p = np.zeros((self.n_y, m))

        # Forward Propagation
        probs, caches = self.forward_propagation(X, self.parameters)

        # Convert probabilities to 0/1 predictions
        for i in range(0, probs.shape[1]):
            for j in range(0, probs.shape[0]):
                if probs[j, i] > 0.5:
                    p[j, i] = 1
                else:
                    p[j, i] = 0

        return p

    def training(self, X, Y, parameters, print_cost):
        costs = []

        # Loop (gradient descent)
        for i in tqdm(range(0, self.num_iterations)):
            # Forward propagation
            AL, caches = self.forward_propagation(X, parameters)

            # Compute cost
            cost = self.compute_cost(AL, Y)

            # Backward propagation
            grads = self.backpropagation(AL, Y, caches)

            # Update weights of the network
            parameters = self.update_weights(parameters, grads, self.learning_rate)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

            costs.append(cost)

        return parameters, costs

    def optimized_training(self, X, Y, parameters, print_cost):
        costs = []

        v, s = self.initialize_adam(parameters)

        m = X.shape[1]
        mini_batch_size = 256
        t = 0  # initializing the counter required for Adam update

        for i in tqdm(range(0, self.num_iterations)):
            minibatches = self.random_mini_batches(X, Y, mini_batch_size)
            cost_total = 0

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # Forward propagation
                AL, caches = self.forward_propagation(minibatch_X, parameters)

                # Compute cost and add to the cost total
                cost_total += self.compute_cost(AL, minibatch_Y)

                # Backward propagation
                grads = self.backpropagation(AL, minibatch_Y, caches)

                # Update parameters
                t = t + 1  # Adam counter
                parameters, v, s = self.update_parameters_with_adam(parameters, grads, v, s, t)

            cost_avg = cost_total / m

            # Print the cost every 1000 epoch
            if print_cost and i % 1000 == 0:
                print("Cost after epoch %i: %f" % (i, cost_avg))
            costs.append(cost_avg)

        return parameters, costs

    def random_mini_batches(self, X, Y, mini_batch_size=64):
        m = X.shape[1]
        n_y = Y.shape[0]
        mini_batches = []

        np.random.seed(0)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((n_y, m))

        num_complete_minibatches = math.floor(m / mini_batch_size)

        for k in range(0, num_complete_minibatches):
            start = k * mini_batch_size
            end = (k + 1) * mini_batch_size

            mini_batch_X = shuffled_X[:, start:end]
            mini_batch_Y = shuffled_Y[:, start:end]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        if m % mini_batch_size != 0:
            start = end
            end = end + m - mini_batch_size * num_complete_minibatches

            mini_batch_X = shuffled_X[:, start:end]
            mini_batch_Y = shuffled_Y[:, start:end]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def initialize_adam(self, parameters):
        L = len(parameters) // 2  # number of layers in the neural networks
        v = {}
        s = {}

        # Initialize v, s. Input: "parameters". Outputs: "v, s".
        for l in range(L):
            v["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
            v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)
            s["dW" + str(l + 1)] = np.zeros(parameters["W" + str(l + 1)].shape)
            s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l + 1)].shape)

        return v, s

    def update_parameters_with_adam(self, parameters, grads, v, s, t):
        epsilon = 1e-8
        L = len(parameters) // 2  # number of layers in the neural networks
        v_corrected = {}  # Initializing first moment estimate, python dictionary
        s_corrected = {}  # Initializing second moment estimate, python dictionary

        # Perform Adam update on all parameters
        for l in range(L):
            # Moving average of the gradients.
            v["dW" + str(l + 1)] = self.beta1 * v["dW" + str(l + 1)] + (1 - self.beta1) * grads['dW' + str(l + 1)]
            v["db" + str(l + 1)] = self.beta1 * v["db" + str(l + 1)] + (1 - self.beta1) * grads['db' + str(l + 1)]

            # Compute bias-corrected first moment estimate
            v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(self.beta1, t))
            v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(self.beta1, t))

            # Moving average of the squared gradients
            s["dW" + str(l + 1)] = self.beta2 * s["dW" + str(l + 1)] + (1 - self.beta2) * np.power(grads['dW' + str(l + 1)], 2)
            s["db" + str(l + 1)] = self.beta2 * s["db" + str(l + 1)] + (1 - self.beta2) * np.power(grads['db' + str(l + 1)], 2)

            # Compute bias-corrected second raw moment estimate
            s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(self.beta2, t))
            s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(self.beta2, t))

            # Update parameters
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - self.learning_rate * (v_corrected["dW" + str(l + 1)] / (np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon))
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - self.learning_rate * (v_corrected["db" + str(l + 1)] / (np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon))

        return parameters, v, s

    def load_params(self):
        parameters = np.load("./outputs/dnn-params.npy", allow_pickle=True)

        self.parameters = parameters.item()

    def leaky_relu(self, Z):
        A = np.maximum(Z, 0.01*Z)

        assert (A.shape == Z.shape)

        cache = Z
        return A, cache

    def leaky_relu_derivative(self, dA, cache):
        Z = cache
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

        # When z <= 0, you should set dz to 0 as well.
        dZ[Z <= 0] = 0.01

        assert (dZ.shape == Z.shape)

        return dZ

    def sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        cache = Z

        return A, cache

    def sigmoid_derivative(self, dA, cache):
        Z = cache

        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)

        assert (dZ.shape == Z.shape)

        return dZ

    def softmax(self, Z):
        shift = Z - np.max(Z)
        t = np.exp(shift)
        A = t / np.sum(t)

        cache = Z

        return A, cache

    def initialize_parameters(self, layer_dims):
        parameters = {}
        L = len(layer_dims)  # number of layers in the network
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2. / layer_dims[l - 1])
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

            assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
            assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

        return parameters

    def linear_forward(self, A, W, b):
        Z = W.dot(A) + b

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        Z, linear_cache = self.linear_forward(A_prev, W, b)

        if activation == "sigmoid":
            A, activation_cache = self.sigmoid(Z)
        elif activation == "relu":
            A, activation_cache = self.leaky_relu(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))

        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_propagation(self, X, parameters):
        caches = []
        A = X
        L = len(parameters) // 2  # number of layers in the neural network

        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation="relu")
            caches.append(cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
        caches.append(cache)

        assert (AL.shape == (self.n_y, X.shape[1]))

        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]

        # Prevents infinite log
        AL[AL == 0.] = 0.001
        AL[AL == 1.] = 0.999

        # Compute loss from aL and y.
        cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))  # Sigmoid

        cost = np.squeeze(np.mean(np.mean(cost, axis=1), axis=0))

        return cost

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1. / m * np.dot(dZ, A_prev.T)
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = self.leaky_relu_derivative(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_derivative(dA, activation_cache)

        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def backpropagation(self, AL, Y, caches):
        grads = {}
        L = len(caches)  # the number of layers
        Y = Y.reshape(AL.shape)

        # Initializing the Backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        current_cache = caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation="sigmoid")

        for l in reversed(range(L - 1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def update_weights(self, parameters, grads, learning_rate):
        L = len(parameters) // 2  # number of layers in the neural network

        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

        return parameters

    def plot_cost(self):
        plt.plot(self.costs)
        plt.ylabel('Cost')
        plt.xlabel('Epochs')
        plt.show()

    def get_accuracy(self, predictions, labels):
        acc = predictions.T - labels
        acc = np.mean(acc, axis=1, keepdims=True)

        acc = acc[acc == 0]
        acc = acc.shape[0] / labels.shape[0] * 100

        return acc
