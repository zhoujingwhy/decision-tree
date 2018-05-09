# -*- coding: UTF-8 -*-
from math import log

"""
函数说明:创建测试数据集
Parameters:
    无
Returns:
    dataSet - 数据集
    labels - 分类属性
Author:
    Csc
Modify:
    2018-04-18
"""
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],   #数据集
            [0, 0, 0, 1, 'no'],      #年龄，有工作，有房子，信贷情况，是否给贷款
            [0, 1, 0, 1, 'yes'],     #年龄：0青年 1中年 2 老年
            [0, 1, 1, 0, 'yes'],     #工作：0否 1是
            [0, 0, 0, 0, 'no'],      #房子：0否 1是
            [1, 0, 0, 0, 'no'],      #信贷情况：0一般 1好 2非常好
            [1, 0, 0, 1, 'no'],      #是否给贷款：no否 yes 是
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']        #分类属性
    return dataSet, labels                #返回数据集和分类属性

"""
函数说明:计算给定数据集的香农熵

Parameters:
    dataSet - 数据集
Returns:
    shannonEnt - 经验熵农熵
Modify:
    2018-04-18
"""
def calcShannonEnt(dataSet):
    numEntires = len(dataSet)                       #返回数据集的行数
    labelCounts = {}                                #保存每个标签(Label)出现次数的字典
    for featVec in dataSet:                         #对每组特征向量进行统计
        currentLabel = featVec[-1]                  #提取标签(Label)信息
        if currentLabel not in labelCounts.keys():  #如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1                #Label计数
    shannonEnt = 0.0                                #经验熵(香农熵)
    for key in labelCounts:                            #计算香农熵
        prob = float(labelCounts[key]) / numEntires    #选择该标签(Label)的概率
        shannonEnt -= prob * log(prob, 2)            #利用公式计算
    return shannonEnt                                #返回经验熵(香农熵)

"""
函数说明:按照给定特征划分数据集

Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
Returns:
    无
"""
def splitDataSet(dataSet, axis, value):
    retDataSet = []                                  #创建返回的数据集列表
    for featVec in dataSet:                          #遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]          #去掉axis特征
            reducedFeatVec.extend(featVec[axis+1:])  #将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet                                #返回划分后的数据集
"""
ID3算法：核心是在决策树各个结点上对应信息增益准则选择特征
函数说明:选择最优特征
Parameters:
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值

"""
def chooseBestFeatureToSplitID3(dataSet):
    numFeatures = len(dataSet[0]) - 1                   #特征数量
    baseEntropy = calcShannonEnt(dataSet)               #计算数据集的香农熵
    bestInfoGain = 0.0                                  #信息增益
    bestFeature = -1                                    #最优特征的索引值
    for i in range(numFeatures):                        #遍历所有特征
        #获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                      #创建set集合{},元素不可重复
        newEntropy = 0.0                                #经验条件熵
        for value in uniqueVals:                        #计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)     #subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))     #计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  #根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy                  #信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))      #打印每个特征的信息增益
        if (infoGain > bestInfoGain):                        #计算信息增益
            bestInfoGain = infoGain                          #更新信息增益，找到最大的信息增益
            bestFeature = i                                  #记录信息增益最大的特征的索引值
    return bestFeature                                       #返回信息增益最大的特征的索引值
'''
C4.5算法：核心是在决策树各个结点上对应信息增益率准则选择特征
函数说明:选择最优特征
Parameters:
    dataSet - 数据集
Returns:
    bestFeature - 信息增益最大的(最优)特征的索引值
'''
def chooseBestFeatureToSplitC45(dataSet):
    numFeatures = len(dataSet[0]) - 1      # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
    bestInfoGain = 0.0                     # 信息增益
    bestFeature = -1                       # 最优特征的索引值
    for i in range(numFeatures):  # 遍历所有特征
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建set集合{},元素不可重复
        newEntropy = 0.0            # 经验条件熵
        splitInfo=0.0
        for value in uniqueVals:    # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)    # subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))    # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet) # 根据公式计算经验条件熵
            splitInfo -=prob * log(prob,2)
        infoGain = (baseEntropy - newEntropy )/splitInfo # 信息增益率
        print("第%d个特征的增益率为%.3f" % (i, infoGain))# 打印每个特征的信息增益率
        if (infoGain > bestInfoGain):  # 计算信息增益
            bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature  # 返回信息增益最大的特征的索引值


if __name__ == '__main__':
    dataSet, features = createDataSet()
    print("ID3算法得出的最优特征索引值:" + str(chooseBestFeatureToSplitID3(dataSet)))
    print("C4.5算法得出的最优特征索引值：" +str(chooseBestFeatureToSplitC45(dataSet)))