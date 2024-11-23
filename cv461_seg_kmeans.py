"""
在图像处理中，通过 K-Means 聚类算法可以实现图像分割、图像聚类、
图像识别等操作，本小节主要用来进行图像颜色分割。假设存在一张 100×100
像素的灰度图像，它由 10000 个 RGB 灰度级组成，我们通过 K-Means 可
以将这些像素点聚类成 K 个簇，然后使用每个簇内的质心点来替换簇内所有的
像素点，这样就能实现在不改变分辨率的情况下量化压缩图像颜色，实现图像颜色层级分割。

在 OpenCV 中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data-1, K, bestLabels,
criteria, attempts, flags[, centers])
 data-1 表示聚类数据，最好是 np.flloat32 类型的 N 维点集
 K 表示聚类类簇数
 bestLabels 表示输出的整数数组，用于存储每个样本的聚类标签
索引
 criteria 表示算法终止条件，即最大迭代次数或所需精度。在某些
迭代中，一旦每个簇中心的移动小于 criteria.epsilon，算法就会
停止
 attempts 表示重复试验 kmeans 算法的次数，算法返回产生最
佳紧凑性的标签
 flags 表示初始中心的选择，两种方法是
cv2.KMEANS_PP_CENTERS ; 和
cv2.KMEANS_RANDOM_CENTERS
 centers 表示集群中心的输出矩阵，每个集群中心为一行数据
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans_1():
    # 读取原始图像灰度颜色
    img = cv2.imread('res/sc_th.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取图像高度、宽度
    rows, cols = img.shape[:]
    # 图像二维像素转换为一维
    data = img.reshape((rows * cols, 1))
    data = np.float32(data)
    # 定义中心 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS
    # K-Means 聚类 聚集成 4 类
    compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
    # 生成最终图像
    dst = labels.reshape((img.shape[0], img.shape[1]))
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图像
    titles = ['原始图像', '聚类图像']
    images = [img, dst]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

def kmeans_2():
    # 读取原始图像
    img = cv2.imread('res/sc_th.jpg')
    # 图像二维像素转换为一维
    data = img.reshape((-1, 3))
    data = np.float32(data)
    # 定义中心 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS
    # K-Means 聚类 聚集成 2 类
    compactness, labels2, centers2 = cv2.kmeans(data, 2,None, criteria, 10, flags)
    # K-Means 聚类 聚集成 4 类
    compactness, labels4, centers4 = cv2.kmeans(data, 4,None, criteria, 10, flags)
    # K-Means 聚类 聚集成 8 类
    compactness, labels8, centers8 = cv2.kmeans(data, 8,None, criteria, 10, flags)
    # K-Means 聚类 聚集成 16 类
    compactness, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)
    # K-Means 聚类 聚集成 64 类
    compactness, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)
    # 图像转换回 uint8 二维类型
    centers2 = np.uint8(centers2)
    res = centers2[labels2.flatten()]
    dst2 = res.reshape((img.shape))
    centers4 = np.uint8(centers4)
    res = centers4[labels4.flatten()]
    dst4 = res.reshape((img.shape))
    centers8 = np.uint8(centers8)
    res = centers8[labels8.flatten()]
    dst8 = res.reshape((img.shape))
    centers16 = np.uint8(centers16)
    res = centers16[labels16.flatten()]
    dst16 = res.reshape((img.shape))
    centers64 = np.uint8(centers64)
    res = centers64[labels64.flatten()]
    dst64 = res.reshape((img.shape))

    # 图像转换为 RGB 显示
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
    dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
    dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
    dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
    dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图像
    titles = ['原始图像', '聚类图像 K=2', '聚类图像 K=4',
              '聚类图像 K=8', '聚类图像 K=16', '聚类图像 K=64']
    images = [img, dst2, dst4, dst8, dst16, dst64]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray'),
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# kmeans_1()
kmeans_2()