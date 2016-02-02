#By: Will Spurgeon and Daniel Pongratz

import numpy
import math

class Neuron:

    def __init__(self, inputs, hiddenLayerNum):
        self.hiddenLayerNum = hiddenLayerNum
        self.inputs = inputs
        self.output = 0.00000
        self.inputWeights  = []
        self.netValue = 0.0000
        self.bias = 0.0000
        self.delta = 0.0000
        self.learningRate = 0.5
        self.errorFactor = 0.00000

        #generate weights for inputs.
        for input in inputs:
            num = numpy.random.random()
            self.inputWeights.append(num)

            #generate bias
            biasNum = numpy.random.random()
            #print "Bias: ", biasNum
            self.bias = biasNum

    def findNetValue(self):
        i = 0
        self.netValue = 0.0
        for input in self.inputs:
            self.netValue = self.netValue + float(input.output) * self.inputWeights[i]
            #print "NetValue ", self.netValue
            i = i + 1
        return self.netValue

    def getOutput(self):
        value = self.findNetValue()
        #print "Value",  value
        self.output = 1/(1+math.exp(-value))
        #print "Output", self.output
        return self.output

    def errorFactorOfHiddenNeuron(self, outputNode):
        self.errorFactor = self.errorFactor + (outputNode.delta * outputNode.inputWeights[self.hiddenLayerNum])
        return self.errorFactor

    def getDelta(self, expectedOutput):
        self.delta = self.output * (1-self.output) * (float(expectedOutput) - self.output)
        return self.delta

    def updateParameters(self):
        self.bias = self.bias + (self.learningRate * 1 * self.delta)
        i = 0
        for inputConnection in self.inputWeights:
            self.inputWeights[i] = self.inputWeights[i] + (self.learningRate * float(self.inputs[i].output) * float(self.delta))
            i = i + 1

