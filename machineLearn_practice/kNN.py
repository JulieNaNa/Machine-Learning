import numpy as np, operator, matplotlib.pyplot as plt,array,os

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    fr = open(filename)
    for line in fr.readlines():
        listFromLine = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    fr.close()
    return (
     returnMat, classLabelVector)

#----------------test data to be used to test algorithm-----------------#
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['a', 'a', 'b', 'b']
    return (
     group, labels)
#------------------------------------------------------------------------#


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat =np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    sortedDistances = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistances[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=1)
    return sortedClassCount[0][0]


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet -np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return (normDataSet, ranges, minVals)

def datingClassTest():
	hoRatio=0.08
	datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals = autoNorm(datingDataMat)
	m = normMat.shape[0] #total numbers of dataset
	numTestVecs = int(m*hoRatio)# 10% of total numbers for testing the classifier
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:],normMat[numTestVecs
:m,:],datingLabels[numTestVecs:m],3)
		print ("the classifier came back with :%d,the real answer is:%d"% (classifierResult,datingLabels[i]))
		if classifierResult != datingLabels[i]:
			errorCount +=1.0
			print '   *********************'
	print 'the error rate is : %f'% (errorCount/float(numTestVecs))


#--------------DATE interface with user-------------------#
def classifyPerson():
	resultList=['not at all','in small doses','in large doses']
	persentTime=float(raw_input("persent of time spent playing video games?"))
	freFliMiles=float(raw_input("frequent flier miles earned per year?"))
	iceCream=float(raw_input("liters of ice cream consumed per year?"))
	datingDataMat,datingLabels=file2matrix("datingTestSet2.txt")
	normMat,ranges,minVals=autoNorm(datingDataMat)
	inArr=array.array('f',[freFliMiles,persentTime,iceCream])
	classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
	print ("you will probably like this persion:",resultList[classifierResult-1])

#----------------------------img2vector-----------------------#
def img2vector(filename):
	returnVect = np.zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

#--------------------handwritten digits testing code------------#
dir1=os.getcwd()
dir0=dir1+'/data/Ch02/'
trainDir=dir0 + 'trainingDigits'
testDir = dir0 + 'testDigits'
	
def dataFolder2dataMat(dirPath):
	hwLabels = []
	dataFileList = os.listdir(dirPath)
	m = len(dataFileList)
	dataMat = np.zeros((m,1024))
	for i in range(m):
		fileNameStr=dataFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr = fileStr.split('_')[0]
		hwLabels.append(int(classNumStr))
		dataMat[i,:]=img2vector('%s/%s' % (dirPath,dataFileList[i]))
	return dataMat,hwLabels

def handwritingClassTest(trainDir=trainDir,testDir=testDir):
	trainingMat,trainingLabels = dataFolder2dataMat(trainDir)
	testingMat,testingRealLabels  = dataFolder2dataMat(testDir)
	errorCount = 0.0
	for i in range(np.shape(testingMat)[0]):
		classifierResult = classify0(testingMat[i,:],trainingMat,trainingLabels,3)
		print "the classifier came with : %d,the real answer is: %d" %(classifierResult,testingRealLabels[i])
		if classifierResult != testingRealLabels[i]: errorCount += 1.0
	errorRate = errorCount/np.shape(testingMat)[0]
	print 'total eror counts is %d'% errorCount
	print 'error rate is %f'% errorRate
	return errorRate



	
	

			





