# -*- coding:utf-8 -*-

import cv2
import numpy as np

# 下面代码是图像平移的一个简单案例，它定义了图像平移矩阵 M，然后调用
# warpAffine()函数将原始图像垂直向下平移了 50 个像素，水平向右平移了
# 100 个像素

#读取图片
src = cv2.imread('res/sc_th.jpg')
#图像平移矩阵
M = np.float32([[1, 0, 100], [0, 1, 50]])
#获取原始图像列数和行数
rows, cols = src.shape[:2]
#图像平移
result = cv2.warpAffine(src, M, (cols, rows))
#显示图像
cv2.imshow("original", src)
cv2.imshow("result", result)
#等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()



