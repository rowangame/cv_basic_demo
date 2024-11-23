"""
 灰度平均值：该值等于图像中所有像素灰度值之和除以图像的像素个数。
 灰度中值：对图像中所有像素灰度值进行排序，然后获取所有像素最中
间的值，即为灰度中值。
 灰度标准差：又常称均方差，是离均差平方的算术平均数的平方根。标
准差能反映一个数据集的离散程度，是总体各单位标准值与其平均数离
差平方的算术平均数的平方根。如果一幅图看起来灰蒙蒙的， 那灰度标
准差就小；如果一幅图看起来很鲜艳，那对比度就很大，标准差也大
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#函数: 获取图像的灰度平均值
def fun_mean(img, height, width):
    sum_img = 0
    for i in range(height):
        for j in range(width):
            sum_img = sum_img + int(img[i, j])
    mean = sum_img / (height * width)
    return mean

#函数: 获取中位数
def fun_median(data):
    length = len(data)
    data.sort()
    if (length % 2) == 1:
        z = length // 2
        y = data[z]
    else:
        y = (int(data[length // 2]) + int(data[length // 2 - 1])) / 2
    return y

img = cv2.imread("res/lena_256_gray.png")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height = grayImg.shape[0]
width = grayImg.shape[1]

#计算图像的灰度平均值
mean = fun_mean(grayImg, height, width)
print("灰度平均值：", mean)

#计算图像的灰度中位数
value = grayImg.ravel() #获取所有像素值
median = fun_median(value)
print("灰度中值：", median)

#计算图像的灰度标准差
std = np.std(value, ddof = 1)
print("灰度标准差", std)

