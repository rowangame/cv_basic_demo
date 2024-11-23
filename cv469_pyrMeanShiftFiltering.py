"""
1.基于均值漂移算法的图像分割
均值漂移（Mean Shfit）算法是一种通用的聚类算法，最早是 1975 年
Fukunaga 等人在一篇关于概率密度梯度函数的估计论文中提出[1]。它是一种
无参估计算法，沿着概率梯度的上升方向寻找分布的峰值。Mean Shift 算法先
算出当前点的偏移均值，移动该点到其偏移均值，然后以此为新的起始点，继续
移动，直到满足一定的条件结束。
图像分割中可以利用均值漂移算法的特性，实现彩色图像分割。在
OpenCV 中提供的函数为 pyrMeanShiftFiltering()，该函数严格来说并不
是图像分割，而是图像在色彩层面的平滑滤波，它可以中和色彩分布相近的颜色，
平滑色彩细节，侵蚀掉面积较小的颜色区域，所以在 OpenCV 中它的后缀是滤
波“Filter”，而不是分割“segment”

该函数原型如下所示：
dst = pyrMeanShiftFiltering(src, sp, sr[, dst[, maxLevel[,
termcrit]]])
 src 表示输入图像，8 位三通道的彩色图像
 dst 表示输出图像，需同输入图像具有相同的大小和类型
 sp 表示定义漂移物理空间半径的大小
 sr 表示定义漂移色彩空间半径的大小
 maxLevel 表示定义金字塔的最大层数
 termcrit 表示定义的漂移迭代终止条件，可以设置为迭代次数满足
终止，迭代目标与中心点偏差满足终止，或者两者的结合

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def pyrMeanShiftFiltering_1():
    #读取原始图像灰度颜色
    img = cv2.imread('res/sc_th.jpg')
    spatialRad = 50 #空间窗口大小
    colorRad = 50 #色彩窗口大小
    maxPyrLevel = 2 #金字塔层数
    #图像均值漂移分割
    dst = cv2.pyrMeanShiftFiltering( img, spatialRad, colorRad, maxPyrLevel)
    #显示图像
    cv2.imshow('src', img)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

# 为了达到更好地分割目的，借助漫水填充函数进行下一步处理
def pyrMeanShiftFiltering_2():
    # 读取原始图像灰度颜色
    img = cv2.imread('res/sc_th.jpg')
    # 获取图像行和列
    rows, cols = img.shape[:2]
    # mask 必须行和列都加 2 且必须为 uint8 单通道阵列
    mask = np.zeros([rows + 2, cols + 2], np.uint8)
    spatialRad = 50  # 空间窗口大小
    colorRad = 50  # 色彩窗口大小
    maxPyrLevel = 2  # 金字塔层数
    # 图像均值漂移分割
    dst = cv2.pyrMeanShiftFiltering(img, spatialRad, colorRad, maxPyrLevel)
    # 图像漫水填充处理
    cv2.floodFill(dst, mask, (30, 30), (0, 255, 255),
                  (100, 100, 100), (50, 50, 50),
                  cv2.FLOODFILL_FIXED_RANGE)

    #显示图像
    cv2.imshow('src', img)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

# pyrMeanShiftFiltering_1()
pyrMeanShiftFiltering_2()
