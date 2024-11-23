"""
形态学理论知识
数学形态学的应用可以简化图像数据，保持它们基本的形状特征，并出去不
相干的结构。数学形态学的算法有天然的并行实现的结构，主要针对的是二值图
像（0 或 1）。在图像处理方面，二值形态学经常应用到对图像进行分割、细化、
抽取骨架、边缘提取、形状分析、角点检测，分水岭算法等。由于其算法简单，
算法能够并行运算所以经常应用到硬件中

常见的图像形态学运算包括：
 腐蚀
 膨胀
 开运算
 闭运算
 梯度运算
 顶帽运算
 底帽运算
这些运算在 OpenCV 中主要通过 MorphologyEx()函数实现，它能利用
基本的膨胀和腐蚀技术，来执行更加高级形态学变换，如开闭运算、形态学梯度、
顶帽、黑帽等，也可以实现最基本的图像膨胀和腐蚀。其函数原型如下：

dst = cv2.morphologyEx(src, model, kernel)
 src 表示原始图像
 model 表示图像进行形态学处理，包括：
(1) cv2.MORPH_OPEN：开运算（Opening Operation）
(2)cv2.MORPH_CLOSE：闭运算（Closing Operation）
(3)cv2.MORPH_GRADIENT：形态学梯度（Morphological
Gradient）
(4)cv2.MORPH_TOPHAT：顶帽运算（Top Hat）
(5)cv2.MORPH_BLACKHAT：黑帽运算（Black Hat）
 kernel 表示卷积核，可以用 numpy.ones()函数构建
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#图像腐蚀处理
# dst = cv2.erode(src, kernel, iterations)
#  src 表示原始图像
#  kernel 表示卷积核
#  iterations 表示迭代次数，默认值为 1，表示进行一次腐蚀操作
def erode():
    # 读取图片
    src = cv2.imread('res/ns_rect.png', cv2.IMREAD_UNCHANGED)
    # 设置卷积核
    kernel = np.ones((5, 5), np.uint8)
    # 图像腐蚀处理
    erosion = cv2.erode(src, kernel)
    # 显示图像
    cv2.imshow("src", src)
    cv2.imshow("result", erosion)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 图像膨胀操作
# 图像膨胀是腐蚀操作的逆操作，类似于“领域扩张”，它将图像中的高亮区
# 域或白色部分进行扩张，其运行结果比原图的高亮区域更大
# 设 A，B 为集合，∅为空集，A 被 B 的膨胀，记为 A⊕B，其中⊕为膨胀算
# 子，膨胀定义为：
# 该公式表示用 B 来对图像 A 进行膨胀处理，其中 B 是一个卷积模板，其形
# 状可以为正方形或圆形，通过模板 B 与图像 A 进行卷积计算，扫描图像中的每
# 一个像素点，用模板元素与二值图像元素做“与”运算，如果都为 0，那么目标
# 像素点为 0，否则为 1。从而计算 B 覆盖区域的像素点最大值，并用该值替换参
# 考点的像素值实现图像膨胀。图 15-4 是将左边的原始图像 A 膨胀处理为右边
def dilate():
    # 读取图片
    src = cv2.imread('res/bar_zhiwen.jpg', cv2.IMREAD_UNCHANGED)
    # grayImag = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # src = grayImag

    # 设置卷积核
    kernel = np.ones((5, 5), np.uint8)
    # 图像膨胀处理
    erosion = cv2.dilate(src, kernel)
    # 显示图像
    cv2.imshow("src", src)
    cv2.imshow("result", erosion)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testAll():
    # 读取图片
    src = cv2.imread('res/ns_pic_j.png', cv2.IMREAD_UNCHANGED)
    # 设置卷积核
    kernel = np.ones((5, 5), np.uint8)

    # 图像腐蚀处理
    erodeImg = cv2.erode(src, kernel)

    # 图像膨胀处理
    dilateImg = cv2.dilate(src, kernel)

    # 设置字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图像
    titles = ['原图', '图像腐蚀', '图像膨胀']
    images = [src, erodeImg, dilateImg]
    for i in range(3):
        plt.subplot(1, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# erode()
# dilate()
testAll()


