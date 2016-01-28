import sys


def main():
    filename = sys.argv[1]
    hiddenNodesNum = sys.argv[2]
    holdoutPercentage = sys.argv[3]

    print filename
    print hiddenNodesNum
    print holdoutPercentage


if __name__ == "__main__":
    main()