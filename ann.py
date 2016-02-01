import sys
from DataPoint import DataPoint
import numpy
import math

#Two input nodes in the input layer
#h number of hidden nodes in the hidden layer
#Two output node in the output layer.

W1 = 1


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


    numpy.random.seed(42)
    W1 = numpy.random.randn(2, numOfHiddenNodes) / numpy.sqrt(2)
    B1 = numpy.zeros((1, numOfHiddenNodes))
    W2 = numpy.random.randn(numOfHiddenNodes, 2) / numpy.sqrt(numOfHiddenNodes)
    B2 = numpy.zeros((1, 2))

    for i in xrange(0, numberOfPasses):
        for example in input:
            for inputNode in range(0,2):
                #For each input node.



    for i in xrange(0, numOfHiddenNodes):
        z1 = input.dot(W1) + B1
        a1 = numpy.tanh(z1)
        z2 = a1.dot(W2) + B2
        score = numpy.exp(z2)
        probs = score / numpy.sum(score, axis=1, keepdims=True) #Fix this.



def classifyPoint(self, dataPoint):

if __name__ == "__main__":
    main()