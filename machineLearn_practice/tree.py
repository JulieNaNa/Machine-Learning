from math import log
import random
import numpy as np
#-------------------calculate the the overall entropy----------------#
def calcEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts:
			labelCounts[currentLabel] = 1
		else:labelCounts[currentLabel] += 1
	entropy = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		entropy -= prob*log(prob,2)
	return entropy

#-------------------testData-----------#
def createDataSet():
	dataSet = [
		[1,1,'maybe'],
		[1,1,'yes'],
		[1,0,'yes'],
		[0,1,'maybe'],
		[0,0,'no']]
	return dataSet

#-----------------split dataSet based on one feature------------#
def splitData(dataSet,axis,value):
	leftDataSet = []
	rightDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reduceFeatVec = featVec[:axis]
			reduceFeatVec.extend(featVec[axis+1:])
			leftDataSet.append(reduceFeatVec)
		else:
			reduceFeatVec = featVec[:axis]
			reduceFeatVec.extend(featVec[axis+1:])
			rightDataSet.append(reduceFeatVec)
	return leftDataSet,rightDataSet

#----check if the subDataSet is terminal-------------------------#
def isTerminal(dataSet):
	labels = {}
	for featVec in dataSet:
		#print featVec[-1],type(featVec[-1])
		
		if featVec[-1] not in labels:
			labels[featVec[-1]] = 1
		else:
			labels[featVec[-1]] +=1
	#print labels
	if len(labels) > 1 and max(labels.values())>1:
		return 0
	return 1

#--------------recurse all the branches-----------------------#
def splitRecurse(dataSet):
	dataSet = dataSet
	n = len(dataSet[0])-2
	#print n
	axis = random.randint(0,n)
	value = dataSet[0][axis]	
	subDataSet = splitData(dataSet,axis,value)[0]
	subResult = isTerminal(subDataSet)
	print subDataSet,subResult
	if subResult == 0:
		splitRecurse(subDataSet)
	subDataSet = splitData(dataSet,axis,value)[1] 	
	subResult = isTerminal(subDataSet)
	print subDataSet,subResult
	if subResult == 0:
		splitRecurse(subDataSet)
	else: 
		return 1

#-----calculate the entropy based on feature selection order--------#

dataSet = createDataSet()
print calcEnt(dataSet)
print dataSet
print splitRecurse(dataSet)



