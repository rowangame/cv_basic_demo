# -*- coding:utf-8 -*-

"""
图像采样（Image Sampling）处理是将一幅连续图像在空间上分割成
M×N 个网格，每个网格用一个亮度值或灰度值来表示，其示意图如图 9-1 所
示。
图像采样的间隔越大，所得图像像素数越少，空间分辨率越低，图像质量越
差，甚至出现马赛克效应；相反，图像采样的间隔越小，所得图像像素数越多，
空间分辨率越高，图像质量越好，但数据量会相应的增大。图 9-2 展示了不同
采样间隔的“Lena”图，其中图(a)为原始图像，图(b)为 128×128 的图像采
样效果，图(c)为 64×64 的图像采样效果，图(d)为 32×32 的图像采样效果，
图(e)为 16×16 的图像采样效果，图(f)为 8×8 的图像采样效果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像
img = cv2.imread('res/lena_256.png')
#获取图像高度和宽度
height = img.shape[0]
width = img.shape[1]
#采样转换成 16*16 区域
numHeight = int(height / 16)
numWidth = int(width / 16)
#创建一幅图像
new_img = np.zeros((height, width, 3), np.uint8)
#图像循环采样 16*16 区域
for i in range(16):
    # 获取 Y 坐标
    y = i * numHeight
    for j in range(16):
        # 获取 X 坐标
        x = j * numWidth
        # 获取填充颜色 左上角像素点
        b = img[y, x][0]
        g = img[y, x][1]
        r = img[y, x][2]
        # 循环设置小区域采样
        for n in range(numHeight):
            for m in range(numWidth):
                new_img[y + n, x + m][0] = np.uint8(b)
                new_img[y + n, x + m][1] = np.uint8(g)
                new_img[y + n, x + m][2] = np.uint8(r)
#显示图像
cv2.imshow("src-1", img)
cv2.imshow("Sampling", new_img)
#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()
