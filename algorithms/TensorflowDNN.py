#!/usr/bin/env python
# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class TensorflowDNN:
    def __init__(self, layers_dims, n_x, n_y, learning_rate=0.0075, num_iterations=3000, beta1=0.9, beta2=0.999):
        self.n_x = n_x  # Number of features
        self.n_y = n_y  # Number of output units
        self.layers_dims = layers_dims  # Number of units in each layer
        self.learning_rate = learning_rate  # Learning rate for weight update
        self.num_iterations = num_iterations
        self.beta1 = beta1
        self.beta2 = beta2
        self.parameters = []
        self.costs = []

    def fit(self, X_train, Y_train, optimizer='adam', print_cost=False):
        ops.reset_default_graph()
        tf.set_random_seed(1)
        seed = 3
        costs = []
        minibatch_size = 256

        # Create Input and Output Placeholders
        X, Y = self.create_placeholders()

        # Initialize parameters
        parameters = self.initialize_parameters()

        # Forward propagation: Build the forward propagation in the tensorflow graph
        ZL = self.forward_propagation(X, parameters)

        # Cost function: Add cost function to tensorflow graph
        cost = self.compute_cost(ZL, Y)

        # Backpropagation: Define the tensorflow optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        # Initialize all the variables
        init = tf.global_variables_initializer()

        # Start the session to compute the tensorflow graph
        with tf.Session() as sess:

            # Run the initialization
            sess.run(init)

            # Do the training loop
            for epoch in tqdm(range(0, self.num_iterations)):

                epoch_cost = 0.  # Defines a cost related to an epoch
                seed = seed + 1
                minibatches = self.random_mini_batches(X_train, Y_train, minibatch_size, seed)

                for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                    epoch_cost += minibatch_cost / minibatch_size

                # Print the cost every epoch
                if print_cost is True and epoch % 100 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                costs.append(epoch_cost)

            self.costs = costs
            self.parameters = sess.run(parameters)
            np.save("./outputs/tf-dnn-params.npy", self.parameters)

    def predict(self, X_pred):
        m = X_pred.shape[1]
        p = np.zeros((self.n_y, m))

        X, _ = self.create_placeholders()

        with tf.Session() as sess:
            # Forward Propagation
            prop = self.forward_propagation(X, self.parameters)
            probs = sess.run(prop, feed_dict={X: X_pred})

        # Convert probabilities to 0/1 predictions
        for i in range(0, probs.shape[1]):
            for j in range(0, probs.shape[0]):
                if probs[j, i] > 0.5:
                    p[j, i] = 1
                else:
                    p[j, i] = 0

        return p

    def load_params(self):
        path = "./outputs/tf-dnn-params.npy"
        parameters = np.load(path, allow_pickle=True)

        self.parameters = parameters.item()

    def create_placeholders(self):
        X = tf.placeholder(shape=(self.n_x, None), dtype=tf.float32, name='X')
        Y = tf.placeholder(shape=(self.n_y, None), dtype=tf.float32, name='Y')

        return X, Y

    def initialize_parameters(self):
        weights_shape = []
        for l in range(len(self.layers_dims) - 1):
            w = [self.layers_dims[l+1], self.layers_dims[l]]
            b = [self.layers_dims[l+1], 1]
            weights_shape.append(w)
            weights_shape.append(b)

        L = len(weights_shape) // 2

        parameters = {}
        weight_idx = 0
        for l in range(0, L):
            w_label = 'W' + str(l + 1)
            b_label = 'b' + str(l + 1)

            parameters[w_label] = tf.get_variable(w_label, weights_shape[l + weight_idx], initializer=tf.contrib.layers.xavier_initializer())
            parameters[b_label] = tf.get_variable(b_label, weights_shape[l + weight_idx + 1], initializer=tf.zeros_initializer())

            weight_idx += 1

        return parameters

    def forward_propagation(self, X, parameters):
        L = len(parameters) // 2

        linears = {}
        activations = {}
        activations['A0'] = X

        for l in range(0, L):
            linears['Z' + str(l + 1)] = tf.matmul(parameters['W' + str(l + 1)], activations['A' + str(l)]) + parameters['b' + str(l + 1)]

            if l + 1 != L:
                z = linears['Z' + str(l + 1)]
                activations['A' + str(l + 1)] = z * tf.math.tanh(tf.math.softplus(z))

        return linears['Z' + str(L)]

    def compute_cost(self, ZL, Y):
        logits = tf.transpose(ZL)
        labels = tf.transpose(Y)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        return cost

    def random_mini_batches(self, X, Y, mini_batch_size=64, seed=0):
        m = X.shape[1]
        n_y = Y.shape[0]
        mini_batches = []

        np.random.seed(seed)
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

    def plot_cost(self):
        plt.plot(np.squeeze(self.costs))
        plt.ylabel('Cost')
        plt.xlabel('Epochs')
        plt.title("Learning Rate =" + str(self.learning_rate))
        plt.show()

    def get_accuracy(self, X_pred, Y_pred, type=''):
        X, Y = self.create_placeholders()

        X_pred = X_pred.T
        Y_pred = Y_pred.T

        with tf.Session() as sess:
            # Forward Propagation
            prop = self.forward_propagation(X, self.parameters)

            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(prop), tf.argmax(Y))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy = accuracy.eval({X: X_pred, Y: Y_pred}) * 100

            print('    -', type, "Accuracy:", accuracy, '%')
