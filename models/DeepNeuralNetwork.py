#!/usr/bin/env python
# coding: utf-8

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(27)


class DeepNeuralNetwork:
    def __init__(self, layers_dims, n_x, n_y, learning_rate=0.0075, num_iterations=3000):
        self.n_x = n_x                          # Number of features
        self.n_y = n_y                          # Number of output units
        self.layers_dims = layers_dims          # Number of units in each layer
        self.learning_rate = learning_rate      # Learning rate for weight update
        self.num_iterations = num_iterations
        self.parameters = []
        self.costs = []

    def fit(self, X, Y, print_cost=False):
        costs = []

        # Parameters initialization.
        parameters = self.initialize_parameters(self.layers_dims)

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
            if i % 100 == 0:
                costs.append(cost)

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
        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = self.sigmoid(Z)

        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self.linear_forward(A_prev, W, b)
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

        assert(m > 0)

        # Prevents infinite log
        AL[AL == 0.] = 0.001
        AL[AL == 1.] = 0.999

        # Compute loss from aL and y.
        cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))

        cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

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
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

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
