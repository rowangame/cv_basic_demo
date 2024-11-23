# -*- coding:utf-8 -*-

# 图像旋转变换主要调用 getRotationMatrix2D()函数和 warpAffine()函
# 数实现，绕图像的中心旋转，函数原型如下
#
# M = cv2.getRotationMatrix2D(center, angle, scale)
#
# center 表示旋转中心点，通常设置为(cols/2, rows/2)
#  angle 表示旋转角度，正值表示逆时针旋转，坐标原点被定为左上角
#  scale 表示比例因子
# rotated = cv2.warpAffine(src, M, (cols, rows))
#  src 表示原始图像
#  M 表示旋转参数，即 getRotationMatrix2D()函数定义的结果
#  (cols, rows)表示原始图像的宽度和高度
import math
import time

import cv2
import numpy as np

# 对图像进行旋转且会裁剪图
def test1():
    #读取图片
    src = cv2.imread('res/sc_th.jpg')
    #源图像的高、宽 以及通道数
    rows, cols, channel = src.shape

    #绕图像的中心旋转
    #函数参数：旋转中心 旋转度数 scale
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)

    #函数参数：原始图像 旋转参数 元素图像宽高
    max_value = max(cols, rows)
    rotated = cv2.warpAffine(src, M, (cols, rows))

    #显示图像
    cv2.imshow("src", src)
    cv2.imshow("rotated", rotated)
    #等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test2():
    #读取图片
    src = cv2.imread('res/sc_th.jpg')
    result = rotate_bound(src, 30)
    #显示图像
    cv2.imshow("src", src)
    cv2.imshow("rotated", result)
    #等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test3():
    height = 4
    width = 2
    new_img = np.zeros((height, width, 3), np.uint8)
    print(new_img.shape)
    print(type(new_img))
    print(new_img)

def test4():
    #读取图片
    src = cv2.imread('res/sc_th.jpg')
    result2 = rotate_bound(src, -30)
    result3 = rotate_bound_ex(src, 30)
    #显示图像
    cv2.imshow("src", src)
    cv2.imshow("rotated2", result2)
    cv2.imshow("rotated3", result3)
    #等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 最近在做深度学习时需要用到图像处理相关的操作，在度娘上找到的图片旋转方法千篇一律，旋转完成的图片都不是原始大小，
# 很苦恼，于是google到歪果仁的网站扒拉了一个方法，亲测好用，再次嫌弃天下文章一大抄的现象，
# 虽然我也是抄歪果仁的。废话不多说了，直接贴代码了
# https://blog.csdn.net/hui3909/article/details/78854387
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

# 先填充，再旋转
def rotate_bound_ex(image, angle):
    height,width,channel = image.shape
    new_img = image
    if height != width:
        max_value = math.ceil(math.sqrt(height ** 2 + width ** 2))
        # 创建一幅图像(宽高相等)
        new_img = np.zeros((max_value, max_value, 3), np.uint8)
        deltaW = int(math.ceil((max_value - width) / 2))
        deltaH = int(math.ceil((max_value - height) / 2))

        # 赋值图像数据(方法1)
        # for tmpR in range(0, height):
        #     for tmpC in range(0, width):
        #         new_img[tmpR + deltaH, tmpC + deltaW] = image[tmpR, tmpC]

        # 赋值图像数据(方法2)
        # new_img[deltaH:deltaH+height,deltaW:deltaW+width] = image[0:height,0:width]

        # 赋值图像数据(方法3,对单个通道进行赋值)
        new_img[deltaH:deltaH + height, deltaW:deltaW + width, 1] = image[0:height, 0:width, 1]

    # 执行旋转逻辑
    w, h = new_img.shape[:2]
    #函数参数：旋转中心 旋转度数 scale
    M = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    rotated = cv2.warpAffine(new_img, M, (max_value, max_value))
    return rotated

# 放转动画显示(按q键退出)
def vedioShow():
    fps = 24
    dtime = int(1000 / fps)
    #读取图片
    src = cv2.imread('res/sc_th.jpg')
    tmpDegree = 0
    while True:
        tmpDegree = tmpDegree + 5
        img_new = rotate_bound_ex(src, tmpDegree)
        cv2.imshow("gif-show", img_new)
        key = cv2.waitKey(dtime)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # test4()
    vedioShow()