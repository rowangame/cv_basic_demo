# -*- coding: utf-8 -*-

"""
K-Means 聚类实现量化处理
除了通过对像素进行统计比较量化处理，还可以根据像素之间的相似性进行
聚类处理。这里补充一个基于 K-Means 聚类算法的量化处理过程，它能够将
彩色图像 RGB 像素点进行颜色分割和颜色量化。此外，该部分只是带领读者简
单认识该方法，更多 K-Means 聚类的知识将在图像分割文章中进行详细叙述
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def test1():
    # 读取原始图像
    img = cv2.imread('res/lena_256_gray.png')
    # 图像二维像素转换为一维
    data = img.reshape((-1, 3))
    data = np.float32(data)
    # 定义中心 (type,max_iter,epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 设置标签
    flags = cv2.KMEANS_RANDOM_CENTERS
    # K-Means 聚类 聚集成 4 类
    # 它通过 K-Means 聚类算法将彩色人物图像的灰度聚集成八种颜色
    compactness, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, flags)
    # 图像转换回 uint8 二维类型
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    dst = res.reshape((img.shape))
    # 图像转换为 RGB 显示
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图像
    titles = ['原始图像', '聚类量化 K=8']
    images = [img, dst]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    test1()