"""
直方图被广泛应用于计算机视觉领域，在使用边缘和颜色确定物体边界时，
通过直方图能更好地选择边界阈值，进行阈值化处理。同时，直方图对物体与背
景有较强对比的景物的分割特别有用，可以应用于检测视频中场景的变换及图像
中的兴趣点

在 OpenCV 中可以使用
calcHist()函数计算直方图，计算完成之后采用 OpenCV 中的绘图函数，如绘
制矩形的 rectangle() 函 数 ， 绘 制 线 段 的 line() 函 数 来 完 成 。 其 中 ，
cv2.calcHist()的函数原型及常见六个参数如下:

hist = cv2.calcHist(images, channels, mask, histSize,
ranges, accumulate)
 hist 表示直方图，返回一个二维数组
 images 表示输入的原始图像
 channels 表示指定通道，通道编号需要使用中括号，输入图像是
灰度图像时，它的值为[0]，彩色图像则为[0]、[1]、[2]，分别表示
蓝色（B）、绿色（G）、红色（R）
 mask 表示可选的操作掩码。如果要统计整幅图像的直方图，则该
值为 None；如果要统计图像的某一部分直方图时，需要掩码来计
算
 histSize 表示灰度级的个数，需要使用中括号，比如[256]
 ranges 表示像素值范围，比如[0, 255]
 accumulate 表示累计叠加标识，默认为 false，如果被设置为
true，则直方图在开始分配时不会被清零，该参数允许从多个对象
中计算单个直方图，或者用于实时更新直方图；多个直方图的累积
结果用于对一组图像的直方图计算
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def calHist1():
    src = cv2.imread("res/lena_128_gray.png")
    # 计算 256 灰度级的图像直方图
    hist = cv2.calcHist([src], [0], None, [256], [0,255])
    #输出直方图大小、形状、数量
    print(hist.size)
    print(hist.shape)
    # print(hist)
    matplotlib.rcParams['font.sans-serif']=['SimHei']

    # 设置显示区域大小
    plt.figure(figsize=(8,6))
    # 显示原始图像和绘制的直方图
    plt.subplot(121)
    plt.imshow(src, 'gray')
    plt.axis('off')
    plt.title("(a)Lena 灰度图像(h=%d,w=%d)" % (src.shape[0], src.shape[1]))

    plt.subplot(122)
    plt.plot(hist, "-.", color='r')
    plt.xlabel("x-(0-255灰度值)")
    plt.ylabel("y-(灰度值计数)")
    plt.title("(b)直方图曲线")
    # 显示绘图
    plt.show()

# 彩色图像调用 OpenCV 绘制直方图的算法与灰度图像一样，只是从 B、
# G、R 三个放量分别进行计算及绘制
def calHist2():
    src = cv2.imread("res/lena_128.png")
    img_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    # 计算直方图
    histb = cv2.calcHist([src], [0], None, [256], [0, 255])
    histg = cv2.calcHist([src], [1], None, [256], [0, 255])
    histr = cv2.calcHist([src], [2], None, [256], [0, 255])
    # 设置字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # 显示原始图像和绘制的直方图
    plt.subplot(121)
    plt.imshow(img_rgb, 'gray')
    plt.axis('off')
    plt.title("(a)Lena 原始图像")
    plt.subplot(122)
    plt.plot(histb, color='b')
    plt.plot(histg, color='g')
    plt.plot(histr, color='r')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("(b)直方图曲线")
    plt.show()

calHist1()
# calHist2()