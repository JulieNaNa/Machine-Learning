#!/usr/bin/env python
#-*- encoding:utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.append('../lib')
#import pudb;pu.db
from PLA import PerceptronPrimary as pla
pla = pla()	
path = pla.filePath()
print path
dataSet = pla.getRawData(path)[0]
TrainSet,TestSet = pla.holdOut(dataSet)
featureTrain,labelsTrain = pla.AnalyzeData(TrainSet)
pla.iniParameter(featureTrain)
pla.updateP(featureTrain,labelsTrain)
print '法向量为：' 
print pla.w
print '截距为：'
print pla.b
labelsTest,labesTestPredict = pla.predict(TestSet)
print 'real Labels:'
print labelsTest
print 'predict Labels:'
print labesTestPredict
PerformanceIndex = pla.evalue(labelsTest,labesTestPredict)
print 'PerformanceIndex:'
print PerformanceIndex
pla.plot(TestSet)
