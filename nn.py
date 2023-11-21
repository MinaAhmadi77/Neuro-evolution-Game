import numpy as np
from numpy import random


class NeuralNetwork():

    def __init__(self, layer_sizes):
        # layer_sizes example: [4, 10, 2]
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        # TODO (Implement FCNNs architecture here)
        self.num_layers = len(layer_sizes)
        self.sizes = layer_sizes
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        pass

    def activation(self, x):
        """
        The activation function of our neural network, e.g., Sigmoid, ReLU.
        :param x: Vector of a layer in our network.
        :return: Vector after applying activation function.
        """
        # TODO (Implement activation function here)
        return (1.0 / (1.0 + np.exp(-x)))

    def forward(self, x):
        # print("wiegths1 .{} input .{} bias .{}".format(self.weights1.shape, x.shape, self.bias1.shape))
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        # TODO (Implement forward function here)

        for b, w in zip(self.biases, self.weights):
            x = self.activation(np.dot(w, x) + b)
        return x
        pass

    def apply_mutation(self):
        mu = 0
        sigma = 1
        sigma2 = 0.8

        self.weights1 += random.normal(mu, sigma, size=(self.layer_sizes[1], self.layer_sizes[0]))
        self.weights2 += random.normal(mu, sigma, size=(self.layer_sizes[1], self.layer_sizes[1]))
        self.weights3 += random.normal(mu, sigma, size=(self.layer_sizes[2], self.layer_sizes[1]))
        self.bias1 += random.normal(mu, sigma2, size=(self.layer_sizes[1], 1))
        self.bias2 += random.normal(mu, sigma2, size=(self.layer_sizes[1], 1))
        self.bias3 += random.normal(mu, sigma2, size=(self.layer_sizes[2], 1))

# ann = NeuralNetwork([6, 20, 1])
# input_ann = np.array([[1], [2], [3], [4], [5], [6]])
# print(input_ann.shape)
# print(input_ann)
# print(ann.forward(input_ann))