
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Scharr 算子又称为 Scharr 滤波器，也是计算 x 或 y 方向上的图像差分，
# 在 OpenCV 中主要是配合 Sobel 算子的运算而存在的
def scharr():
    # 读取图像
    img = cv2.imread('res/lena_256.png')
    lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Scharr 算子
    x = cv2.Scharr(grayImage, cv2.CV_32F, 1, 0)  # X 方向
    y = cv2.Scharr(grayImage, cv2.CV_32F, 0, 1)  # Y 方向
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Scharr = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图形
    titles = ['原始图像', 'Scharr 算子']
    images = [lenna_img, Scharr]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# Cann 算子
# Canny 算法是一种被广泛应用于边缘检测的标
# 准算法，其目标是找到一个最优的边缘检测解或找寻一幅图像中灰度强度变化最
# 强的位置。最优边缘检测主要通过低错误率、高定位性和最小响应三个标准进行评价
def canny():
    # 读取图像
    img = cv2.imread('res/lena_256.png')
    lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波降噪
    gaussian = cv2.GaussianBlur(grayImage, (3, 3), 0)
    # Canny 算子
    Canny = cv2.Canny(gaussian, 50, 150)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图形
    titles = ['原始图像', 'Canny 算子']
    images = [lenna_img, Canny]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# LOG 算子
# LOG（Laplacian of Gaussian）边缘检测算子是 David Courtnay
# Marr 和 Ellen Hildreth 在 1980 年共同提出的，也称为 Marr & Hildreth 算
# 子，它根据图像的信噪比来求检测边缘的最优滤波器。该算法首先对图像做高斯
# 滤波，然后再求其拉普拉斯（Laplacian）二阶导数，根据二阶导数的过零点来
# 检测图像的边界，即通过检测滤波结果的零交叉（Zero crossings）来获得图
# 像或物体的边缘。
# LOG 算子综合考虑了对噪声的抑制和对边缘的检测两个方面，并且把
# Gauss 平滑滤波器和 Laplacian 锐化滤波器结合了起来，先平滑掉噪声，再
# 进行边缘检测，所以效果会更好。 该算子与视觉生理中的数学模型相似，因此
# 在图像处理领域中得到了广泛的应用。它具有抗干扰能力强，边界定位精度高，
# 边缘连续性好，能有效提取对比度弱的边界等特点
# 由于 LOG 算子到中心的距离与位置加权系数的关系曲线像墨西哥草帽的
# 剖面，所以 LOG 算子也叫墨西哥草帽滤波器
def log():
    # 读取图像
    img = cv2.imread('res/lena_256.png')
    lenna_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 先通过高斯滤波降噪
    gaussian = cv2.GaussianBlur(grayImage, (3, 3), 0)
    # 再通过拉普拉斯算子做边缘检测
    dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)
    LOG = cv2.convertScaleAbs(dst)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图形
    titles = ['原始图像', 'LOG 算子']
    images = [lenna_img, LOG]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# scharr()
canny()
# log()