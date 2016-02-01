import sys
from DataPoint import DataPoint
import numpy
import math
import Neuron

#Two input nodes in the input layer
#h number of hidden nodes in the hidden layer
#One output node in the output layer.


def main():
    if len(sys.argv) == 6:
        #All of the parameters.
        filename = sys.argv[1]
        hiddenNodesNum = sys.argv[3]
        holdoutPercentage = sys.argv[5]
    elif len(sys.argv) == 4:
        if sys.argv[2] == "h":
            filename = sys.argv[1]
            hiddenNodesNum = sys.argv[3]
            holdoutPercentage = .2
        else:
            filename = sys.argv[1]
            hiddenNodesNum = 5
            holdoutPercentage = sys.argv[3]
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        hiddenNodesNum = 5
        holdoutPercentage = .2


    input = []

    print filename
    print hiddenNodesNum
    print holdoutPercentage

    file = open(filename, 'r')

    for line in file:
        lineList = line.split(" ")
        newDataPoint = DataPoint(lineList[0], lineList[1], lineList[2])
        input.append(newDataPoint)



def activationFunction(self, x):
    return 1/(1+math.pow(numpy.e, -x))


def buildModel(self, numOfHiddenNodes, numberOfPasses=10000):
    model = {}

    #Build input and hidden neurons
    inputLayer = [Neuron(), Neuron()]
    hiddenLayer = []

    p = 0
    for j in numOfHiddenNodes:
        newNeuron = Neuron(inputLayer, p)
        hiddenLayer.append(newNeuron)
        p = p + 1

    outputLayer = [Neuron(hiddenLayer)]

    for i in xrange(0, numberOfPasses):
        inputLayer[0].output = input[i].xValue
        inputLayer[1].output = input[i].yValue

        for aHiddenNeuron in hiddenLayer:
            aHiddenNeuron.getOutput()

        for anOutputNeuron in outputLayer:
            anOutputNeuron.getOutput

        #Find error factor of output
        outputLayer[0].delta(input[i].dataLabel)

        #Find error factor of each hidden layer
        for aHiddenNeuron in hiddenLayer:
            aHiddenNeuron.errorFactorOfHiddenNeuron(outputLayer[0])

        #Update parameters of hidden nodes and then output node
        for aNode in hiddenLayer:
            aNode.updateParameters()

        for anOutputNeuron in outputLayer:
            anOutputNeuron.updateParameters()

    return [inputLayer, hiddenLayer, outputLayer]

def classifyPoint(self, dataPoint, network):
    inputLayer = network[0]
    hiddenLayer = network[1]
    outputLayer = network[2]

    inputLayer[0].output = dataPoint.xValue
    inputLayer[1].output = dataPoint.yValue

    for aHiddenNeuron in hiddenLayer:
        aHiddenNeuron.getOutput()

    for anOutputNeuron in outputLayer:
        anOutputNeuron.getOutput

    return outputLayer[0].output

if __name__ == "__main__":
    main()