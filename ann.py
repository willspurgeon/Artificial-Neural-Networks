import sys
from DataPoint import DataPoint


def main():
    filename = sys.argv[1]
    hiddenNodesNum = sys.argv[2]
    holdoutPercentage = sys.argv[3]

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