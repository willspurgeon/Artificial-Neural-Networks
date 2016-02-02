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
            self.inputWeights.append(numpy.random.random())

            #generate bias
            self.bias = numpy.random.random()

    def findNetValue(self):
        i = 0
        for input in self.inputs:
            self.netValue = self.netValue + float(input.output) + self.inputWeights[i]
            #print "NetValue ", self.netValue
            i = i + 1
        self.netValue = self.netValue + self.bias
        return self.netValue

    def getOutput(self):
        value = self.findNetValue()
        #print "Value",  value
        #print "Output ", 1/(1+math.pow(numpy.e, -value))
        self.output = 1/(1+math.pow(numpy.e, -value))
        #print "Output", self.output
        return self.output

    def errorFactorOfHiddenNeuron(self, outputNode):
        self.errorFactor = self.errorFactor + (outputNode.delta * outputNode.inputWeights[self.hiddenLayerNum])
        return self.errorFactor

    def test(self):
        i = 1
        #do nothing

    def getDelta(self, expectedOutput):
        self.delta = self.output * (1-self.output) * (float(expectedOutput) - self.output)
        return self.delta

    def updateParameters(self):
        self.bias = self.bias + (self.learningRate * self.delta)
        i = 0
        for inputConnection in self.inputWeights:
            self.inputWeights[i] = self.inputWeights[i] + self.learningRate * float(self.inputs[i].output) * float(self.delta)
            i = i + 1

