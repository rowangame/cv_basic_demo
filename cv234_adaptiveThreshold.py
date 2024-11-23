# 自适应阈值化处理
"""
前面讲解的是固定值阈值化处理方法，而当同一幅图像上的不同部分具有不
同亮度时，上述方法就不在适用。此时需要采用自适应阈值化处理方法，根据图
像上的每一个小区域，计算与其对应的阈值，从而使得同一幅图像上的不同区域
采用不同的阈值，在亮度不同的情况下得到更好的结果。自适应阈值化处理在
OpenCV 中调用 cv2.adaptiveThreshold()函数实现，其原型如下所示
dst = adaptiveThreshold(src, maxValue, adaptiveMethod,thresholdType, blockSize, C[, dst])
src 表示输入图像
 dst 表示输出的阈值化处理后的图像，其类型和尺寸需与 src 一致
 maxValue 表示给像素赋的满足条件的最大值
 adaptiveMethod 表示要适用的自适应阈值算法，常见取值包括
ADAPTIVE_THRESH_MEAN_C（阈值取邻域的平均值） 或
ADAPTIVE_THRESH_GAUSSIAN_C（阈值取自邻域的加权
和平均值，权重分布为一个高斯函数分布）
 thresholdType 表 示 阈 值 类 型 ， 取 值 必 须 为
THRESH_BINARY 或 THRESH_BINARY_INV
 blockSize 表示计算阈值的像素邻域大小，取值为 3、5、7 等
 C 表示一个常数，阈值等于平均值或者加权平均值减去这个常数
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def adaptviteThreshold():
    # 读取图像
    img = cv2.imread('res/bar_miao_clothing.jpg')
    # 图像灰度化处理
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 固定值阈值化处理
    r, thresh1 = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

    # 自适应阈值化处理 方法一
    thresh2 = cv2.adaptiveThreshold(grayImage, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY,
                                    11, 2)

    # 自适应阈值化处理 方法二
    thresh3 = cv2.adaptiveThreshold(grayImage, 255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,
                                    11, 2)
    # 设置字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图像
    titles = ['灰度图像', '全局阈值', '自适应平均阈值', '自适应高斯阈值']
    images = [grayImage, thresh1, thresh2, thresh3]
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

adaptviteThreshold()


