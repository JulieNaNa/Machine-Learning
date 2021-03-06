import random,matplotlib.pyplot as plt
dataSet=[[0,1,2,3,4,5,6,7,8,9,'a'],
	 [10,11,12,13,14,15,16,17,18,19,'b'],
	 [20,21,22,23,24,25,26,27,28,29,'c'],
	 [30,31,32,33,34,35,36,37,38,39,'d'],
         [40,41,42,43,44,45,46,47,48,49,'a'],
         [50,51,52,53,54,55,56,57,58,59,'b'],
         [60,61,62,63,64,65,66,67,68,69,'c'],
         [70,71,72,73,74,75,76,77,78,79,'d'],
         [80,81,82,83,84,85,86,87,88,89,'a'],
         [90,91,92,93,94,95,96,97,98,99,'b']]
 
def splitData(dataSet,axis,value):
        leftDataSet = []
	for featVec in dataSet:
		if featVec[axis] != value:
			reduceFeatVec = featVec[:axis]
                        reduceFeatVec.extend(featVec[axis+1:])
                        leftDataSet.append(reduceFeatVec)
        return leftDataSet
def recurse(dataSet,n):
	a = []
	b=[]
	c=[]
	while n >= 0 :
		axis = random.randint(0,n)
		value = dataSet[0][axis]
		c.append(value)
		a.append(value%10)
		temResult = splitData(dataSet,axis,value)
		dataSet = temResult
		b.append(axis)
		n -= 1
	return b,c,a
result = recurse(dataSet,9)
print result
plt.plot(result[2],result[0])
plt.xlabel('a')
plt.ylabel('b')
plt.show()

