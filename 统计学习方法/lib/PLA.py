#-*- encoding:utf-8 -*-

'''
    感知机模型：f(x) = sign(w*x+b)，属于二分类模型。
    输入空间：长度为N的特征向量空间
    输出空间：{-1,1}
    模型参数：超平面法向量/权值向量：w, 偏置：b
    下面分别从统计学习三要素和学习步骤进行描述
    1：三要素
        假设空间：定义于特征空间中的所有线性分类器，即函数集{f|f(x)= w*x+b}
        模型选择准则：损失函数最小
        模型学习算法：随机梯度下降法
    2：学习步骤
        （1）：样本获取：采样IRIS作为样本,取前两类样本，用户输入文件路径--->选择留出法或k交叉验证法分离训练集和测试集
        （2）：确定模型假设空间为{f|f(x)= w*x+b}
        （3）：确定模型选择准则
            经验风险最小：损失函数为所有误分类点分类器的距离之和，min-∑y(w*x+b)，（x,y）∈M,M为误分类集 
    （4）:确定模型学习算法，梯度下降法		
        （5）：学习算法实现，包括原始算法和对偶算法
        （6）：预测分析
    Author：Julie                                                             Date：20181201                                                           Version: V1.0
'''
import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import numpy as np
import sympy
import matplotlib.pyplot as plt
import sys
from mpl_toolkits.mplot3d import Axes3D #绘制三维图形


class PerceptronDual(object):
    pass 

class PerceptronPrimary(object):
    ''' 
    目的：利用训练集训练模型，确定w,b，并利用测试集进行评价，并绘图显示
    属性：w,b
    方法：获取特征矩阵、更新参数、评价、绘图
    '''
    def __init__(self,w=np.array([0,]),b=0,r=1):
        self.w = w
        self.b = b
        self.r = r
    def filePath(self):
        path = raw_input('请输入文件名(仅是文件名)：\n')
        dirpath =os.path.dirname(sys.path[0])
        path = os.path.join(dirpath,'data',path)
        return path
    def getRawData(self,path):
        '''
        目的：从原始文件中读入前100个样本，包含两类样本，每个样本有四个属性
        由于数据已按类别按顺序存放，为提取训练集和测试集，更改了索引顺序
        参数：数据文件路径，字符串
        返回：已打乱顺序的包含两类的样本总体，正类集，负类集
        '''
        df = pd.read_csv(path,header=None) #读取样本，返回一个异构类型的二维数组,类型为DataFrame
        number = 100
        columns=[1,2,3,4,'label']
        df.columns = columns
        df = df.iloc[0:number]
        df = df.replace(to_replace = {'Iris-setosa': 1, 'Iris-versicolor': -1})#花名为Iris-setosa，则为正类，否则为负类
        index = range(number)
        for i in  range(int(number/2)):
            index[2*i] = i
            index[2*i+1] = i+50
        dataSet = df.reindex(index = index)#打乱顺序的样本总体
        dataP = df.iloc[0:int(number/2)-1].values#所有正类
        dataN = df.iloc[int(number/2):number-1].values#所有负类
        return dataSet,dataP,dataN	
    def holdOut(self,dataSet):
        '''
        目的：用留出法从样本总体中分离出测试集和训练集
        参数：样本总体，训练集占样本总体的比列
        返回：测试集、训练集
        '''
        ratio = float(input('请输入采用留出法分离训练集和测试集时所用比例')) 
        number = dataSet.shape[0]
        numberTrain = int(number*ratio)
        TrainSet = dataSet[0:numberTrain]
        TestSet = dataSet[numberTrain:100]
        return TrainSet,TestSet

    def kCrossValidation(self,dataSet):
        '''
        目的：用K折交叉验证法从样本总体中分离出测试集和训练集
        参数：样本总体，子集个数
        返回：测试集、训练集
        '''
        k = int(input('请输入采用交叉验证分离训练集和测试集时用的子集个数')) 
        number = dataSet.shape[0]
        columns = dataSet.shape[1]
        subNumber = int(number/k)
        subSets = np.empty((k,subNumber,columns))
        TrainSet = np.zeros(((k-1)*subNumber,columns))
        for i in range(k):
            subSets[i]= dataSet[i*subNumber:(i+1)*subNumber]
            TestSet  = subSets[k-1]
            TrainSet = np.delete(subSets,k-1,axis = 0).reshape(((k-1)*subNumber,columns))
        return subSets,TrainSet,TestSet
    def AnalyzeData(self,dataSet):
        '''
        目的：从样本中分离出特征矩阵和标签
        参数：样本
        返回：特征矩阵、标签
        '''
        featureMatrix = dataSet.iloc[:,0:4]
        labels = pd.DataFrame(dataSet.iloc[:,4],columns = ['label'])
        return featureMatrix,labels
    def iniParameter(self,featureMatrix):
        '''
        目的：根据特征矩阵的维数初始化w为0向量
        参数：特征矩阵
        返回：None
        '''
        number = featureMatrix.shape[1]
        self.w,self.b = np.zeros(number),0 #初始参数为0
        self.r = float(input('请输入学习步长')) 
    def updateP(self,featureMatrix,labels):
        '''
        目的：train the model,update w,b
        参数：训练集，测试集
        返回：超平面，评价指标
        '''
        labesPredict = np.dot(featureMatrix,self.w)+self.b   #求出预测值
        Y = labesPredict*labels['label']		  #预测值和实际类别是否一致
        errIndex = np.where(Y<=0)[0] #求出yi(w*xi+b)<=0的索引，包括等于0的意思是最后并没有样本落在超平面上 
        if np.size(errIndex)==0:
            pass
        else:
            x = featureMatrix.iloc[errIndex[0],:]
            y = labels.iloc[errIndex[0]][0]
            self.w = self.w+np.dot(x,y)*self.r
            self.b = self.b+self.r*y
            return self.updateP(featureMatrix,labels) 

    def predict(self,TestSet):
        '''
        目的：预测测试集样本类别
        参数：测试集，超平面参数
        返回：真实类别、预测类别
        '''
        featureMatrix,labels  = self.AnalyzeData(TestSet)
        index = labels.index
        columns = labels.columns
        labesPredict =  np.dot(featureMatrix,self.w)+self.b
        labesPredict = pd.DataFrame(data = labesPredict,index = index,columns = columns)
        labesPredict.loc[labesPredict['label'] > 0] = 1
        labesPredict.loc[labesPredict['label'] < 0] = -1
        labesPredict.loc[labesPredict['label'] == 0] = 0
        return labels,labesPredict

    def evalue(self,labels,labesPredict):
        '''
        目的：评价模型
        参数：真实类别、预测类别
        返回：各种评价指标，以字典形式返回
        '''
        P_real = set(np.where(labels == 1)[0])
        N_real  = set(np.where(labels == -1)[0])
        P_predict = set(np.where(labesPredict == 1)[0])
        N_predict  = set(np.where(labesPredict == -1)[0])
        TP = float(len(P_real&P_predict))
        FN = float(len(P_real&N_predict))
        FP = float(len(N_real&P_predict))
        TN = float(len(N_real&N_predict))
        TPR = TP/(FN+TP)
        FNR = 1-TPR
        FPR = FP/(FP+TN)
        TNR = 1-FPR 
        ACC = (TP+TN)/(FN+TP+FP+TN)
        Ave_Acc = (TPR+TNR)/2
        Error = 1-ACC
        Precision = TP/(TP+FP)
        Recall = TP/(TP+FN)
        F1_Score = 2*Recall*Precision/(Recall+Precision)
        PerformanceIndex = {'TP':TP,'FN':FN,'FP':FP,'TN':TN,
                            'TPR':TPR,'FNR':FNR,'FPR':FPR,'TNR':TNR,
                            'ACC':ACC,'Ave_Acc':Ave_Acc,'Error':Error,
                            'Precision':Precision,'Recall':Recall,'F1-Score':F1_Score}	
        return PerformanceIndex
    def seperate(self,dataSet):
        positiveSet = dataSet.loc[dataSet['label']== 1]
        negativeSet = dataSet.loc[dataSet['label']== -1]
        return positiveSet,negativeSet
    def axisData3(self,dataSet):
        x = dataSet.iloc[:,0]
        y = dataSet.iloc[:,1]
        z = dataSet.iloc[:,2]
        return x,y,z
    def plot(self,TestSet):
        fig = plt.figure(num = 1,figsize = (14,12),dpi = 140,facecolor = 'white',edgecolor = 'black',frameon = True )
        ax = fig.add_axes([0,0,1,1],projection = '3d')
        #ax = fig.add_subplot(111, projection='3d')
        x1,y1,z1 = self.axisData3(self.seperate(TestSet)[0]) 
        x2,y2,z2 = self.axisData3(self.seperate(TestSet)[1]) 
        ax.scatter(x1,y1,z1,s=300,c='red',marker='>')
        ax.scatter(x2,y2,z2,s=300,c='green', marker='o') 
        x = sympy.Symbol('x')
        y = sympy.Symbol('y')
        z = sympy.Symbol('z')
        w = self.w[0:3]
        expression = sympy.solve((np.dot(w,[x,y,z])+self.b),z)#根据法向量w和截距b求取z的表达式
        expression = expression[0]
        xData = self.seperate(TestSet)[0][1]
        yData = self.seperate(TestSet)[0][2]
        zData = []
        for i in xData:
            zData.append(expression.evalf(subs={x:i,y:i})) #根据表达式求取z
        ax = fig.gca()
        ax.plot(xData,yData,zData,c = 'blue')#画超平面
        plt.title(r'PLA',fontsize=40)#设定图标的显示样式属性等
        plt.xlabel('x(1)',fontsize=30)
        plt.ylabel('x(2)',fontsize=30)
        plt.tick_params(labelsize=23)
        name = raw_input("请输入图像文件名称：")
        dirpath =os.path.dirname(sys.path[0])
        path = os.path.join(dirpath,'result',name)
        plt.savefig(path)
       # plt.show()
        os.system('eog'+' '+path+'.png')
        
