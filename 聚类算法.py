import numpy as np
import matplotlib.pyplot as plt


# 加载数据
def loadDataSet(fileName):
    data = np.loadtxt(fileName, delimiter='\t')
    return data


# 欧氏距离计算
def distEclud(x, y):
    return np.sqrt(np.sum((x - y) ** 2))  # 计算欧氏距离


# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet, k):
    m, n = dataSet.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        index = int(np.random.uniform(0, m))  #
        centroids[i, :] = dataSet[index, :]
    return centroids


# k均值聚类
def KMeans(dataSet, k):
    m = np.shape(dataSet)[0]  # 行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))
    clusterChange = True

    # 第1步 初始化centroids
    centroids = randCent(dataSet, k)
    while clusterChange:
        clusterChange = False

        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1

            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = distEclud(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        # 第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值

    print("Congratulations,cluster complete!")
    return centroids, clusterAssment


def showCluster(dataSet, k, centroids, clusterAssment):
    m, n = dataSet.shape
    if n != 2:
        print("数据不是二维的")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("k值太大了")
        return 1

    # 绘制所有的样本
    for i in range(m):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制质心
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i])

    plt.show()


dataSet = loadDataSet("testSet.txt")
k = 6
centroids, clusterAssment = KMeans(dataSet, k)

showCluster(dataSet, k, centroids, clusterAssment)


# from numpy import *
# # 加载数据
# def loadDataSet(fileName):  # 解析文件，按tab分割字段，得到一个浮点数字类型的矩阵
#     dataMat = []              # 文件的最后一个字段是类别标签
#     fr = open(fileName)
#     for line in fr.readlines():
#         curLine = line.strip().split('\t')
#         fltLine = map(float, curLine)    # 将每个元素转成float类型
#         dataMat.append(fltLine)
#     return dataMat
#
# # 计算欧几里得距离
# def distEclud(vecA, vecB):
#     return sqrt(sum(power(vecA - vecB, 2))) # 求两个向量之间的距离
#
# # 构建聚簇中心，取k个(此例中为4)随机质心
# def randCent(dataSet, k):
#     n = shape(dataSet)[1]
#     centroids = mat(zeros((k,n)))   # 每个质心有n个坐标值，总共要k个质心
#     for j in range(n):
#         minJ = float(min(dataSet[:,j]))
#         maxJ = float(max(dataSet[:,j]))
#         rangeJ = float(maxJ - minJ)
#         centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
#     return centroids
#
# # k-means 聚类算法
# def kMeans(dataSet, k, distMeans =distEclud, createCent = randCent):
#     m = shape(dataSet)[0]
#     clusterAssment = mat(zeros((m,2)))    # 用于存放该样本属于哪类及质心距离
#     # clusterAssment第一列存放该数据所属的中心点，第二列是该数据到中心点的距离
#     centroids = createCent(dataSet, k)
#     clusterChanged = True   # 用来判断聚类是否已经收敛
#     while clusterChanged:
#         clusterChanged = False
#         for i in range(m):  # 把每一个数据点划分到离它最近的中心点
#             minDist = inf; minIndex = -1
#             for j in range(k):
#                 distJI = distMeans(centroids[j,:], dataSet[i,:])
#                 if distJI < minDist:
#                     minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点更近，则将i归属为j
#             if clusterAssment[i,0] != minIndex: clusterChanged = True  # 如果分配发生变化，则需要继续迭代
#             clusterAssment[i,:] = minIndex,minDist**2   # 并将第i个数据点的分配情况存入字典
#         print(centroids)
#         for cent in range(k):   # 重新计算中心点
#             ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   # 去第一列等于cent的所有列
#             centroids[cent,:] = mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
#     return centroids, clusterAssment
# # --------------------测试----------------------------------------------------
# # 用测试数据及测试kmeans算法
# datMat = mat(loadDataSet('testSet.txt'))
# myCentroids,clustAssing = kMeans(datMat,4)
# print(myCentroids)
# print(clustAssing)