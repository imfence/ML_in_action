import os
import numpy as np
import KNN_0


def img2vec(filename):
    returnVec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32*i+j] = int(lineStr[j])
    return returnVec


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vec('trainingDigits/%s' % (fileNameStr))
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vecUnderTest = img2vec('testDigits/%s' % (fileNameStr))
        clfResult = KNN_0.classify0(vecUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with:%d, the ral answer is:%d" %
              (clfResult, classNumStr))
        if(clfResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is:%d" % errorCount)
    print("\nthe total error rate if:%f" % (errorCount/float(mTest)))

handwritingClassTest()
