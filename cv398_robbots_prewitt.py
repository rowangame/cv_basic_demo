
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Roberts 算子又称为交叉微分算法，它是基于交叉差分的梯度算法，通过
局部差分计算检测边缘线条。常用来处理具有陡峭的低噪声图像，当图像边缘接
近于正 45 度或负 45 度时，该算法处理效果更理想，其缺点是对边缘的定位不
太准确，提取的边缘线条较粗
在 Python 中，Roberts 算子主要通过 Numpy 定义模板，再调用
OpenCV 的 filter2D()函数实现边缘提取[3]。该函数主要是利用内核实现对图
像的卷积运算，其函数原型如下所示：
dst = filter2D(src, ddepth, kernel[, dst[, anchor[, delta[,
borderType]]]])
 src 表示输入图像
 dst 表示输出的边缘图，其大小和通道数与输入图像相同
 ddepth 表示目标图像所需的深度
 kernel 表示卷积核，一个单通道浮点型矩阵
 anchor 表示内核的基准点，其默认值为（-1，-1），位于中心位置
 delta 表示在储存目标图像前可选的添加到像素的值，默认值为 0
 borderType 表示边框模式
"""
def robots():
    #读取图像
    img = cv2.imread('res/lena_256.png')
    lenna_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Roberts 算子
    kernelx = np.array([[-1,0],[0,1]], dtype=int)
    kernely = np.array([[0,-1],[1,0]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    #转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Roberts = cv2.addWeighted(absX,0.5,absY,0.5,0)
    #用来正常显示中文标签
    plt.rcParams['font.sans-serif']=['SimHei']

    #显示图形
    titles = ['原始图像', 'Roberts 算子']
    images = [lenna_img, Roberts]
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

# Prewitt 算子
# Prewitt 是一种图像边缘检测的微分算子，其原理是利用特定区域内像素灰
# 度值产生的差分实现边缘检测。由于 Prewitt 算子采用 3×3 模板对区域内的
# 像素值进行计算，而 Robert 算子的模板为 2×2，故 Prewitt 算子的边缘检测
# 结果在水平方向和垂直方向均比 Robert 算子更加明显。Prewitt 算子适合用
# 来识别噪声较多、灰度渐变的图像
def prewitt():
    #读取图像
    img = cv2.imread('res/lena_256.png')
    lenna_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Prewitt 算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)

    # 转 uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图形
    titles = ['原始图像', 'Prewitt 算子']
    images = [lenna_img, Prewitt]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# robots()
prewitt()
