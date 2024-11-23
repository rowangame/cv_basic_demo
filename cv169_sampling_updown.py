# -*- coding: utf-8 -*-

# 在图像向上取样是由小图像不断放图像的过程。它将图像在每个方向上扩大
# 为原图像的 2 倍，新增的行和列均用 0 来填充，并使用与“向下取样”相同的
# 卷积核乘以 4，再与放大后的图像进行卷积运算，以获得“新增像素”的新值。
# 如图 10-2 所示，它在原始像素 45、123、89、149 之间各新增了一行和一列
# 值为 0 的像素

import cv2
import numpy as np
import matplotlib.pyplot as plt

# dst = pyrUp(src[, dst[, dstsize[, borderType]]])
#  src 表示输入图像，
#  dst 表示输出图像，和输入图像具有一样的尺寸和类型
#  dstsize 表示输出图像的大小，默认值为 Size()
#  borderType 表示像素外推方法，详见 cv::bordertypes
def pyrUpTest():
    # 读取原始图像
    img = cv2.imread('res/lena_256.png')
    # 图像向上取样
    r = cv2.pyrUp(img)
    # 显示图像
    cv2.imshow('original', img)
    cv2.imshow('PyrUp', r)
    cv2.waitKey()
    cv2.destroyAllWindows()

def pyrUpManyTest():
    # 读取原始图像
    img = cv2.imread('res/lena_128.png')
    # 图像向上取样
    r1 = cv2.pyrUp(img)
    r2 = cv2.pyrUp(r1)
    r3 = cv2.pyrUp(r2)
    # 显示图像
    cv2.imshow('original', img)
    cv2.imshow('PyrUp1', r1)
    cv2.imshow('PyrUp2', r2)
    cv2.imshow('PyrUp3', r3)
    cv2.waitKey()
    cv2.destroyAllWindows()

# 在图像向下取样中，使用最多的是高斯金字塔。它将对图像 Gi 进行高斯核
# 卷积，并删除原图中所有的偶数行和列，最终缩小图像。其中，高斯核卷积运算
# 就是对整幅图像进行加权平均的过程，每一个像素点的值，都由其本身和邻域内
# 的其他像素值（权重不同）经过加权平均后得到。
def pyrDownTest():
    # 读取原始图像
    img = cv2.imread('res/lena_512.png')
    # 图像向下取样
    r = cv2.pyrDown(img)
    # 显示图像
    cv2.imshow('original', img)
    cv2.imshow('PyrDown', r)
    cv2.waitKey()
    cv2.destroyAllWindows()

# 每次向下取样均为上次的四分之一，并且图像的清晰度会降低
def pyrDownManyTest():
    # 读取原始图像
    img = cv2.imread('res/lena_512.png')
    #图像向下取样
    r1 = cv2.pyrDown(img)
    r2 = cv2.pyrDown(r1)
    r3 = cv2.pyrDown(r2)
    # 显示图像
    cv2.imshow('original', img)
    cv2.imshow('PyrDown1', r1)
    cv2.imshow('PyrDown2', r2)
    cv2.imshow('PyrDown3', r3)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    pyrUpTest()
    # pyrUpManyTest()
    # pyrDownTest()
    # pyrDownManyTest()