
"""
图像融合
图像融合通常是指多张图像的信息进行融合，从而获得信息更丰富的结果，
能够帮助人们观察或计算机处理。图 5-1 是将两张不清晰的图像融合得到更清
晰的效果图。

区别如下[1-3]：
 图像加法：目标图像 = 图像 1 + 图像 2
 图像融合：目标图像 = 图像 1 × 系数 1 + 图像 2 × 系数 2 + 亮度调节量
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

"""在 OpenCV 中，图像融合主要调用 addWeighted()函数实现，其原型如
下。需要注意的是，两张融合图像的像素大小必须一致，参数 gamma 不能省
略。
dst = cv2.addWeighted(scr1, alpha, src2, beta, gamma)
dst = src1 * alpha + src2 * beta + gamma
"""
def testFusion():
    # 读取图片
    src1 = cv2.imread('res/sc_th.jpg')
    src2 = cv2.imread('res/sc_th2.jpg')

    result = cv2.addWeighted(src1, 0.6, src2, 0.6, 0)
    # 显示图像
    cv2.imshow("src1", src1)
    cv2.imshow("src2", src2)
    cv2.imshow("result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
ROI（Region of Interest）表示感兴趣区域，是指从被处理图像以方框、
圆形、椭圆、不规则多边形等方式勾勒出需要处理的区域。可以通过各种算子
（Operator）和函数求得感兴趣 ROI 区域，被广泛应用于热点地图、人脸识
别、图像分割等领域
"""
def testROI():
    # 读取图片
    img = cv2.imread("res/lena_256.png")
    # 定义 200×200 矩阵 3 对应 BGR
    face = np.ones((124, 124, 3))
    # 显示原始图像
    cv2.imshow("Demo", img)
    # 显示 ROI 区域
    face = img[69:193, 82:206]
    cv2.imshow("face", face)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 赋值ROI区域到其它图像上
def testROIEx():
    # 读取图片
    img = cv2.imread("res/sc_th.jpg")
    img2 = cv2.imread("res/sc_th2.jpg")

    # 定义 200×200 矩阵 3 对应 BGR
    roi = np.ones((124, 124, 3))
    # 显示原始图像
    cv2.imshow("t1", img)
    # 显示 ROI 区域
    roi = img[69:193, 82:206]
    cv2.imshow("roi", roi)

    img2[10:134, 10:134] = roi
    cv2.imshow("roiEx", img2)

    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# testFusion()
# testROI()
testROIEx()