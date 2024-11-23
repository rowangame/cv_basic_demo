"""
直方图均衡化算法的详细处理过程如下：
第一步，计算原始图像直方图的概率密度。
第二步，通过灰度变换函数 T 计算新图像灰度级的概率密度
第三步，计算新图像的灰度值
图像均衡化特点：提高图像的亮度和细节
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalizeHist1():
    #读取图片
    # img = cv2.imread("./res/lena_256.png")
    img = cv2.imread("res/sc_scene_2.jpg")
    #灰度转换
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #直方图均衡化处理
    result = cv2.equalizeHist(gray)

    #显示图像
    cv2.imshow("Input", gray)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def equalizeHist2():
    # 读取图片
    img = cv2.imread('res/lena_256.png')
    # 灰度转换
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化处理
    result = cv2.equalizeHist(gray)

    plt.subplot(221)
    plt.imshow(gray, cmap=plt.cm.gray)
    plt.axis("off"),
    plt.title('(a)')

    plt.subplot(222)
    plt.imshow(result, cmap=plt.cm.gray)
    plt.axis("off"),
    plt.title('(b)')

    plt.subplot(223)
    plt.hist(img.ravel(), 256)
    plt.title('(c)')

    plt.subplot(224)
    plt.hist(result.ravel(), 256)
    plt.title('(d)')
    plt.show()

# 如果需要对彩色图片进行全局直方图均衡化处理，则需要分解 RGB 三色通
# 道，分别进行处理后再进行通道合并
def equalizeHist3():
    img = cv2.imread("res/sc_night.jpg")

    b, g, r = cv2.split(img)
    # 彩色图像均衡化 需要分解通道 对每一个通道均衡化
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)

    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    cv2.imshow("Input", img)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 绘制直方图
    plt.figure("Hist")
    # 蓝色分量
    plt.hist(bH.ravel(), bins=256, facecolor='b', edgecolor='b')
    # 绿色分量
    plt.hist(gH.ravel(), bins=256, facecolor='g', edgecolor='g')
    # 红色分量
    plt.hist(rH.ravel(), bins=256, facecolor='r', edgecolor='r')
    plt.xlabel("x-label")
    plt.ylabel("y-label")
    plt.show()

# equalizeHist1()
# equalizeHist2()
equalizeHist3()

