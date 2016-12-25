import numpy as np 
import data_classification_utils
from util import raiseNotDefined
import random

class Perceptron(object):
    def __init__(self, categories, numFeatures):
        """categories: list of strings 
           numFeatures: int"""
        self.categories = categories
        self.numFeatures = numFeatures

        """YOUR CODE HERE"""
        self.weights = np.zeros((len(categories), numFeatures))


    def classify(self, sample):
        """sample: np.array of shape (1, numFeatures)
           returns: category with maximum score, must be from self.categories"""

        """YOUR CODE HERE"""
        result = 0
        resultIndex = 0
        for index in range(self.weights.shape[0]):
          current = np.dot(self.weights[index], sample)
          if index == 0 or current > result:
            result = current
            resultIndex = index
        return self.categories[resultIndex]


    def train(self, samples, labels):
        """samples: np.array of shape (numFeatures, numSamples)
           labels: list of numSamples strings, all of which must exist in self.categories 
           performs the weight updating process for perceptrons by iterating over each sample once."""

        """YOUR CODE HERE"""
        for i in range(len(samples)):
          number = samples[i]
          current = self.classify(number)
          label = labels[i]
          if current != label:
            self.weights[int(current)] -= number
            self.weights[int(label)] += number
