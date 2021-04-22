import numpy as np
import operator


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    '''
    inX: input vector
    dataSet: train set
    labels: label
    k: k nearest neighbors
    '''
    dataSetSize = dataSet.shape[0]  # row size
    diffMat = np.tile(inX, (dataSetSize, 1))-dataSet  # inX minus dataSet
    sqdiffMat = diffMat ** 2
    sqDistances = sqdiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistances = distances.argsort()
    classCount = {}  # record nearest k labels
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]  # key
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    sortedClasscount = sorted(
        classCount.items(), key=lambda x: x[1], reverse=True)  # sort by value
    return sortedClasscount[0][0]


group, labels = createDataSet()
print(classify0([0, 0], group, labels, 3))
