"""
Laplacian 算子
拉普拉斯（Laplacian）算子是 n 维欧几里德空间中的一个二阶微分算子，
常用于图像增强领域和边缘提取。它通过灰度差分计算邻域内的像素，基本流程
是：
 判断图像中心像素灰度值与它周围其他像素的灰度值；
 如果中心像素的灰度更高，则提升中心像素的灰度；
 反之降低中心像素的灰度，从而实现图像锐化操作。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def laplacian():
    # 读取图像
    img = cv2.imread('res/lena_256.png')
    lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 灰度化处理图像
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 拉普拉斯算法
    dst = cv2.Laplacian(grayImg, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图形
    titles = ['原始图像', 'Laplacian 算子']
    images = [lenna_img, Laplacian]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# 边缘检测算法主要是基于图像强度的一阶和二阶导数，但导数通常对噪声很
# 敏感，因此需要采用滤波器来过滤噪声，并调用图像增强或阈值化算法进行处理，
# 最后再进行边缘检测。下面是采用高斯滤波去噪和阈值化处理之后，再进行边缘
# 检测的过程，并对比了四种常见的边缘提取算法。
# Laplacian 算子对噪声比较敏感，由于其算法可能会出现双像素边界，常用来判断边缘像素位于图像的明区或暗区，很少用于边缘检测；
# Robert 算子对陡峭的低噪声图像效果较好，尤其是边缘正负45 度较多的图像，但定位准确率较差；
# Prewitt 算子对灰度渐变的图像边缘提取效果较好，而没有考虑相邻点的距离远近对当前像素点的影响；
# Sobel 算子考虑了综合因素，对噪声较多的图像处理效果更好
def testCommon():
    # 读取图像
    img = cv2.imread('res/lena_256.png')
    lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    gaussianBlur = cv2.GaussianBlur(grayImage, (3, 3), 0)
    # 阈值处理
    ret, binary = cv2.threshold(gaussianBlur, 127, 255, cv2.THRESH_BINARY)
    # Roberts 算子
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
    y = cv2.filter2D(binary, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # Prewitt 算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
    y = cv2.filter2D(binary, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # Sobel 算子
    x = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(binary, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 拉普拉斯算法
    dst = cv2.Laplacian(binary, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)
    # 效果图
    titles = ['Source Image', 'Binary Image', 'Roberts Image',
              'Prewitt Image', 'Sobel Image', 'Laplacian Image']
    images = [lenna_img, binary, Roberts, Prewitt, Sobel, Laplacian]
    for i in np.arange(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# laplacian()
testCommon()