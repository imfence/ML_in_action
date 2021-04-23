import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import KNN_0


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


#datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
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

#normMat, ranges, minVals=autoNorm(datingDataMat)
#print(normMat, '\n',ranges, '\n',minVals)


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        clfResult = KNN_0.classify0(
            normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 4)
        print("The classifier came back with: %d\n, the real answer is: %d" %
              (clfResult, datingLabels[i]))
        if(clfResult != datingLabels[i]):
            errorCount += 1.0
    print("The total error rate is:%f" % (errorCount/float(numTestVecs)))


# datingClassTest()

def clfPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datngDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datngDataMat)
    inArr = np.array([percentTats,ffMiles,iceCream])
    norminArr=(inArr-minVals)/ranges
    clfResult=KNN_0.classify0(norminArr,normMat,datingLabels,3)
    print("you will probably like this person:",resultList[clfResult-1])

clfPerson()
