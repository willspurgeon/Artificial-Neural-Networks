import sys
from DataPoint import DataPoint


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



if __name__ == "__main__":
    main()