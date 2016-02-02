import sys
from DataPoint import DataPoint
import numpy
import math
from Neuron import Neuron

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

    hiddenNodesNum = int(hiddenNodesNum)
    file = open(filename, 'r')

    for line in file:
        lineList = line.split(" ")
        newDataPoint = DataPoint(lineList[0], lineList[1], lineList[2])
        input.append(newDataPoint)

    model = buildModel(hiddenNodesNum, input, 15000)
    print "Error rate: ", testUsingTestData(model)
    #print classifyPoint(DataPoint(-1.300597, 0.962803), model)

def testUsingTestData(network):
    file = open("testData.txt", 'r')
    testInput = []
    numOfDataPoints = 0
    numOfErrors = 0
    for line in file:
        lineList = line.split(" ")
        newDataPoint = DataPoint(lineList[0], lineList[1], lineList[2])
        testInput.append(newDataPoint)
        numOfDataPoints = numOfDataPoints + 1

    for point in testInput:
        pointClass = classifyPoint(point, network)
        print "Point class: " , int(pointClass), "Expected: ", int(float(point.dataLabel))
        if int(pointClass) != int(float(point.dataLabel)):
            numOfErrors = numOfErrors + 1

    return float(numOfErrors)/float(numOfDataPoints)

def buildModel(numOfHiddenNodes, input, numberOfPasses):
    model = {}

    #Build input and hidden neurons
    in1 = Neuron([], -1)
    in2 = Neuron([], -1)
    inputLayer = [in1, in2]
    hiddenLayer = []

    for j in range(0,numOfHiddenNodes):
        newNeuron = Neuron(inputLayer, j)
        hiddenLayer.append(newNeuron)

    outputLayer = [Neuron(hiddenLayer, -1)]

    for j in xrange(0, numberOfPasses):
        for i in xrange(0, len(input)):
            inputLayer[0].output = input[i].xValue
            inputLayer[1].output = input[i].yValue

            for aHiddenNeuron in hiddenLayer:
                aHiddenNeuron.getOutput()

            for anOutputNeuron in outputLayer:
                anOutputNeuron.getOutput

            #Find error factor of output
            outputLayer[0].getDelta(input[i].dataLabel)

            #Find error factor of each hidden layer
            for aHiddenNeuron in hiddenLayer:
                aHiddenNeuron.errorFactorOfHiddenNeuron(outputLayer[0])

            #Update parameters of hidden nodes and then output node
            for aNode in hiddenLayer:
                aNode.updateParameters()

            for anOutputNeuron in outputLayer:
                anOutputNeuron.updateParameters()

    return [inputLayer, hiddenLayer, outputLayer]

def classifyPoint(dataPoint, network):
    inputLayer = network[0]
    hiddenLayer = network[1]
    outputLayer = network[2]

    inputLayer[0].output = dataPoint.xValue
    inputLayer[1].output = dataPoint.yValue

    for aHiddenNeuron in hiddenLayer:
        aHiddenNeuron.getOutput()

    for anOutputNeuron in outputLayer:
        '''
        print anOutputNeuron.inputWeights[0]
        print anOutputNeuron.inputWeights[1]
        print anOutputNeuron.inputWeights[2]
        print anOutputNeuron.inputWeights[3]
        print anOutputNeuron.inputWeights[4]
        print anOutputNeuron.inputWeights[5]
        print anOutputNeuron.inputWeights[6]
        print anOutputNeuron.inputWeights[7]
        '''

        anOutputNeuron.getOutput()

    output = outputLayer[0].output
    return output

if __name__ == "__main__":
    main()