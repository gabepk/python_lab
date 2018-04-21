#!/usr/bin/python

import numpy

class NeuralNetwork:

    # CONSTRUTOR
    def __init__(self):
        # O peso eh um array de 3 floats (de 0 à 1) - ou uma matrix 3x1
        # O valor inicial do peso começa com 3 floats randomicos
        self.peso = numpy.random.random((3, 1))

    def think(self, inputs):
        return self.sigmoid(numpy.dot(inputs, self.peso))

    def sigmoid(self, x):
        return 1/(1 + numpy.exp(-x))

    def train(self, inputs, outputs, num):
        for iteration in range(num):
            output = self.think(inputs)
            error = outputs - output
            adjustment = numpy.dot(inputs.T, error * output * (1 - output))
            self.peso += adjustment


inputs = numpy.array([[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0], [0, 1, 0]])
outputs = numpy.array([[1], [1], [0], [0], [0]])

network = NeuralNetwork()
network.train(inputs, outputs, 10000)

output = network.think(numpy.array([1, 0, 1]))
print(output)
