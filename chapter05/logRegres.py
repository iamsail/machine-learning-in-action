from numpy import *

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        """
        书上这里说是:为了方便计算,该函数还将X0的值设为1.0
        这里其实是有公式可以参考的,回到线性假设的公式,对应的就是那个常数项
        比如常数项是theta,就可以看做theta * x, x为1.0
        书上说方便,就在于,取1不会改变原来的值
        把这个常数项看做X为1.0,这样在后面就可以通过矩阵来进行计算了 
        """
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    """ 梯度上升求解最佳回归系数

    Args:
        dataMatIn: {List} 样本数据,对应假设中的x
        classLabels: {List} 样本数据,对应假设中的函数值h(x)

    Returns:
        weights: 最佳回归系数矩阵
    """
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    # 向目标移动的步长
    alpha = 0.001
    # 迭代次数
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        # 　参考图二公式,这里是梯度上升
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def plotBestFit(weights):
    """ 画出数据集和Logistic回归最佳拟合直线的函数

        Args:
            weights: 最佳回归系数矩阵
    """
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    # 最佳拟合直线
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    """ 随机梯度上升算法

        Args:
            dataMatrix:  样本数据,对应假设中的x
            classLabels:  样本数据,对应假设中的函数值h(x)

        Returns:
            weights: 最佳回归系数
    """
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """ 改进的随机梯度上升算法

        Args:
            dataMatrix:  样本数据,对应假设中的x
            classLabels:  样本数据,对应假设中的函数值h(x)
            numIter: 迭代次数

        Returns:
            weights: 最佳回归系数
    """
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            """
            这是第一处改进地方.alpha在每次迭代时都会调整,这缓解了图(5)上数据波动或者高频波动。
            另外alpha会随着迭代次数不断减小,但永远不会减小到0。    
            必须这样做的原因是为了保证在多次迭代之后新数据仍然具有一定的影响
            如果要处理的问题是动态变化的,可以适当加大上述常数项,来确保新的值获得更大的回归系数
            """
            alpha = 4 / (1.0 + j + i) + 0.01
            """
            这是第二处改进的地方,这里通过随机选取样本来更新回归系数,这种方法将减少周期性波动    
            """
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(list(dataIndex)[randIndex])
    return weights

def classifyVector(inX, weights):
    """ 分类

        Args:
            inX:  特征向量
            weights: 回归系数

        Returns:
            1 or 0
    """
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    print(trainWeights);
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print('the error rate of this test is : %f' % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is : %f ' % (numTests, errorSum / float(numTests)))