# DB = f(DA) = αDA + b
# 该公式中 DB表示灰度线性变换后的灰度值，DA表示变换前输入图像的灰度
# 值，α 和 b 为线性变换方程 f(D)的参数，分别表示斜率和截距[1-4]。
#  当 α=1，b=0 时，保持原始图像
#  当 α=1，b!=0 时，图像所有的灰度值上移或下移
#  当 α=-1，b=255 时，原始图像的灰度值反转
#  当 α>1 时，输出图像的对比度增强

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 该算法将实现图像灰度值的上移，从而提升图像的亮度。
# DB=DA+50
def grayUp():
    img = cv2.imread("res/lena_256.png")
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取图像高度和宽度
    height = grayImage.shape[0]
    width = grayImage.shape[1]
    # 创建一幅图像
    result = np.zeros((height, width), np.uint8)
    # 图像灰度上移变换 DB=DA+50
    for i in range(height):
        for j in range(width):
            if (int(grayImage[i, j] + 50) > 255):
                gray = 255
            else:
                gray = int(grayImage[i, j] + 50)
            result[i, j] = np.uint8(gray)
    # 显示图像
    cv2.imshow("Gray Image", grayImage)
    cv2.imshow("Result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 该算法将增强图像的对比度，Python 实现代码如下所示。
# DB=DA×1.5
def grayStrengthen():
    img = cv2.imread("res/lena_256.png")
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取图像高度和宽度
    height = grayImage.shape[0]
    width = grayImage.shape[1]
    # 创建一幅图像
    result = np.zeros((height, width), np.uint8)
    # 图像对比度增强变换 DB=DA×1.5
    for i in range(height):
        for j in range(width):
            if (int(grayImage[i, j] * 1.5) > 255):
                gray = 255
            else:
                gray = int(grayImage[i, j] * 1.5)
            result[i, j] = np.uint8(gray)

    # 显示图像
    cv2.imshow("Gray Image", grayImage)
    cv2.imshow("Result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 该算法将减弱图像的对比度，Python 实现代码如下所示
# DB=DA×0.8
def grayAttenuate():
    img = cv2.imread("res/lena_256.png")
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取图像高度和宽度
    height = grayImage.shape[0]
    width = grayImage.shape[1]
    # 创建一幅图像
    result = np.zeros((height, width), np.uint8)
    # 图像对比度增强变换 DB=DA×0.8
    for i in range(height):
        for j in range(width):
            gray = int(grayImage[i, j] * 0.5)
            result[i, j] = np.uint8(gray)

    # 显示图像
    cv2.imshow("Gray Image", grayImage)
    cv2.imshow("Result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 图像灰度反色变换
# 反色变换又称为线性灰度求补变换，它是对原图像的像素值进行反转，即黑
# 色变为白色，白色变为黑色的过程。
#  DB=255-DA
def grayReverse():
    img = cv2.imread("res/lena_256.png")
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取图像高度和宽度
    height = grayImage.shape[0]
    width = grayImage.shape[1]
    # 创建一幅图像
    result = np.zeros((height, width), np.uint8)
    # 图像灰度反色变换 DB=255-DA
    for i in range(height):
        for j in range(width):
            gray = 255 - grayImage[i, j]
            result[i, j] = np.uint8(gray)
    # 显示图像
    cv2.imshow("Gray Image", grayImage)
    cv2.imshow("Result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# grayUp()
# grayStrengthen()
grayAttenuate()
# grayReverse()