import numpy
import math

class Neuron:

    def __init__(self, inputs, hiddenLayerNum = -1):
        self.hiddenLayerNum = -1
        self.inputs = inputs
        self.output = 0
        self.inputWeights  = []
        self.netValue = 0
        self.bias = 0
        self.delta = 0
        self.learningRate = 0.5
        #generate weights for inputs.
         #numpy.random.seed(42)
        #W1 = numpy.random.randn(2, numOfHiddenNodes) / numpy.sqrt(2)
        #B1 = numpy.zeros((1, numOfHiddenNodes))
        #W2 = numpy.random.randn(numOfHiddenNodes, 2) / numpy.sqrt(numOfHiddenNodes)
        #B2 = numpy.zeros((1, 2))
        #generate bias

    def findNetValue(self):
        i = 0
        for input in self.inputs:
            self.netValue = self.netValue + input + self.inputWeights[i]
            i = i + 1
        self.netValue = self.netValue + self.bias

    def getOutput(self):
        value = self.findNetValue()
        self.output = 1/(1+math.pow(numpy.e, -value))
        return self.output

    def errorFactorOfHiddenNeuron(self, outputNode):
        self.errorFactor = self.errorFactor + (outputNode.delta * outputNode.inputWeights[self.hiddenLayerNum])
        return  self.errorFactor

    def delta(self, expectedOutput):
        self.delta = self.output * (1-self.output) * (expectedOutput - self.output)
        return self.delta

    def updateParameters(self):
        self.bias = self.bias + (self.learningRate * self.delta)
        i = 0
        for inputConnection in self.inputWeights:
            self.inputWeights[i] = self.inputWeights[i] + self.learningRate * self.inputs[i].output * self.delta
            i = i + 1

