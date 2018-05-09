#-*- coding:utf-8 -*-
import numpy as np

def loadDataSet(fileName):
    """
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        dataMat - 数据矩阵
    Website:
        http://www.cuijiahua.com/
    Modify:
        2017-12-09
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))                    #转化为float类型
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    """
    函数说明:根据特征切分数据集合
    Parameters:
        dataSet - 数据集合
        feature - 带切分的特征
        value - 该特征的值
    Returns:
        mat0 - 切分的数据集合0
        mat1 - 切分的数据集合1
    Website:
        http://www.cuijiahua.com/
    Modify:
        2017-12-12
    """
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0, mat1
def regLeaf(dataSet):
    """
    函数说明:生成叶结点
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的均值
    Website:
        http://www.cuijiahua.com/
    Modify:
        2017-12-12
    """
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    """
    函数说明:误差估计函数
    Parameters:
        dataSet - 数据集合
    Returns:
        目标变量的总方差
    Website:
        http://www.cuijiahua.com/
    Modify:
        2017-12-12
    """
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1,4)):
    """
    函数说明:找到数据的最佳二元切分方式函数
    Parameters:
        dataSet - 数据集合
        leafType - 生成叶结点
        regErr - 误差估计函数
        ops - 用户定义的参数构成的元组
    Returns:
        bestIndex - 最佳切分特征
        bestValue - 最佳特征值
    Website:
        http://www.cuijiahua.com/
    Modify:
        2017-12-12
    """
    import types
    #tolS允许的误差下降值,tolN切分的最少样本数
    tolS = ops[0]; tolN = ops[1]
    #如果当前所有值相等,则退出。(根据set的特性)
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    #统计数据集合的行m和列n
    m, n = np.shape(dataSet)
    #默认最后一个特征为最佳切分特征,计算其误差估计
    S = errType(dataSet)
    #分别为最佳误差,最佳特征切分的索引值,最佳特征值
    bestS = float('inf'); bestIndex = 0; bestValue = 0
    #遍历所有特征列
    for featIndex in range(n - 1):
        #遍历所有特征值
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            #根据特征和特征值切分数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            #如果数据少于tolN,则退出
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            #计算误差估计
            newS = errType(mat0) + errType(mat1)
            #如果误差估计更小,则更新特征索引值和特征值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    #根据最佳的切分特征和特征值切分数据集合
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    #如果切分出的数据集很小则退出
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    #返回最佳切分特征和特征值
    return bestIndex, bestValue

if __name__ == '__main__':
    myDat = loadDataSet('ex00.txt')
    myMat = np.mat(myDat)
    feat, val = chooseBestSplit(myMat, regLeaf, regErr, (1, 4))
    print(feat)
    print(val)