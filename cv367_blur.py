"""
Python 调用 OpenCV 中的 cv2.blur()函数实现均值滤波处理，其函数
原型如下所示，输出的 dst 图像与输入图像 src 具有相同的大小和类型。

dst = blur(src, ksize[, dst[, anchor[, borderType]]])
 src 表示输入图像，它可以有任意数量的通道，但深度应为
CV_8U、CV_16U、CV_16S、CV_32F 或 CV_64F
 ksize 表示模糊内核大小，以（宽度，高度）的形式呈现
 anchor 表示锚点，即被平滑的那个点，其默认值 Point（-1，-
1）表示位于内核的中央，可省略
 borderType 表示边框模式，用于推断图像外部像素的某种边界模
式，默认值为 BORDER_DEFAULT，可省略
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("res/lena_256.png")
source = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#均值滤波(常用内核(3,3)或(5,5))
# 图像均值滤波是通过模糊内核对图像进行平滑处理，由于模糊内核中的每个
# 权重值都相同，故称为均值。该方法在一定程度上消除了原始图像中的噪声，降
# 低了原始图像的对比度，但也存在一定缺陷，它在降低噪声的同时使图像变得模
# 糊，尤其是边缘和细节处，而且模糊内核越大，模糊程度越严重
result = cv2.blur(source, (16, 16))

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
#显示图形
titles = ['原始图像', '均值滤波']
images = [source, result]
for i in range(2):
    plt.subplot(1, 2, i + 1),
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

