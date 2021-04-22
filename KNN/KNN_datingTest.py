import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVec = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()  # delete the beginning and the end of the line, default:'\n','\t',space
        listFromLine = line.split('\t')  # split by '\t'
        returnMat[index, :] = listFromLine[0:3]
        if listFromLine[-1] == 'didntLike':
            classLabelVec.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVec.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVec.append(3)
        index += 1
    return returnMat, classLabelVec


datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
# print(datingDataMat,'\n',datingLabels)

""" fig,ax = plt.subplots(1,1)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*np.array(datingLabels),15.0*np.array(datingLabels))
plt.show() """


def autoNorm(dataSet):
    # normalization
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = (dataSet - np.tile(minVals, (m, 1))) / ranges
    return normDataSet, ranges, minVals

normMat, ranges, minVals=autoNorm(datingDataMat)
#print(normMat, '\n',ranges, '\n',minVals)

def datingClassTest():
    hoRatio = 0.10
    