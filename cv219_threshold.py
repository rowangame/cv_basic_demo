"""
图像阈值化（Binarization）旨在剔除掉图像中一些低于或高于一定值的像
素，从而提取图像中的物体，将图像的背景和噪声区分开来。
灰度化处理后的图像中，每个像素都只有一个灰度值，其大小表示明暗程度。
阈值化处理可以将图像中的像素划分为两类颜色，常见的阈值化算法如公式

Gray(i,j) = 255, Gray(i,j) >= T
Gray(i,j) = 0, Gray(i,j) < T
"""

# dst = cv2.threshold(src, thresh, maxval, type[, dst])
#  src 表示输入图像的数组，8 位或 32 位浮点类型的多通道数
#  dst 表示输出的阈值化处理后的图像，其类型和通道数与 src 一致
#  thresh 表示阈值
#  maxval 表示最大值，当参数阈值类型 type 选 择
# CV_THRESH_BINARY 或 CV_THRESH_BINARY_INV
# 时，该参数为阈值类型的最大值
#  type 表示阈值类型

import cv2
import numpy as np
import matplotlib.pyplot as plt

#像素点的灰度值大于阈值设其灰度值为最大值，小于阈值的像素点灰度值设定为0
def tresh_binary():
    src = cv2.imread('res/lena_256.png')
    # 灰度图像处理
    grayImage = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # 二进制阈值化处理
    r, b = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    # 显示图像
    cv2.imshow("src", src)
    cv2.imshow("result", b)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 大于阈值的像素点的灰度值设定为 0，而小于该阈值的设定为 255。
def tresh_binary_inv():
    src = cv2.imread("res/lena_256.png")
    grayImage = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    r, b = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("src", src)
    cv2.imshow("result", b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 像素点的灰度值小于阈值不改变，反之将像素点的灰度值设定为该阈值
def tresh_trunc():
    src = cv2.imread('res/lena_256.png')
    # 灰度图像处理
    grayImage = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # 二进制阈值化处理
    r, b = cv2.threshold(grayImage, 127, 0, cv2.THRESH_TRUNC)
    # 显示图像
    cv2.imshow("src--", grayImage)
    cv2.imshow("result", b)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 像素点的灰度值大于该阈值的不进行任何改变，小于该阈值其灰度值全部设定为0(参考文档有误)
def tresh_tozero():
    src = cv2.imread('res/lena_256.png')
    # 灰度图像处理
    grayImage = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # 二进制阈值化处理
    r, b = cv2.threshold(grayImage, 127, 0, cv2.THRESH_TOZERO)
    # 显示图像
    cv2.imshow("src--", grayImage)
    cv2.imshow("result", b)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 像素点的灰度值小于该阈值的不进行任何改变，而大于该阈值的部分，其灰度值全部变为 0(参考文档有误)
def tresh_tozero_inv():
    src = cv2.imread('res/lena_256.png')
    # 灰度图像处理
    grayImage = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # 二进制阈值化处理
    r, b = cv2.threshold(grayImage, 127, 0, cv2.THRESH_TOZERO_INV)
    # 显示图像
    cv2.imshow("src", grayImage)
    cv2.imshow("result", b)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testAll():
    # 读取图像
    img = cv2.imread('res/bar_miao_clothing.jpg')
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 阈值化处理
    ret, thresh1 = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(grayImage, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(grayImage, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(grayImage, 127, 255, cv2.THRESH_TOZERO_INV)

    # 显示结果
    titles = ['Gray Image', 'BINARY', 'BINARY_INV', 'TRUNC','TOZERO', 'TOZERO_INV']
    images = [grayImage, thresh1, thresh2, thresh3, thresh4,thresh5]
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# tresh_binary()
# tresh_binary_inv()
tresh_trunc()
# tresh_tozero()
# tresh_tozero_inv()
# testAll()