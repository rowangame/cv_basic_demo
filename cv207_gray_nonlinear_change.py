#图像灰度非线性变换

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 原始图像的灰度值按照 DB=DA×DA/255 的公式进行非线性变换
def nonlinear1():
    img = cv2.imread("res/lena_256.png")
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取图像高度和宽度
    height = grayImage.shape[0]
    width = grayImage.shape[1]
    # 创建一幅图像
    result = np.zeros((height, width), np.uint8)
    # 图像灰度非线性变换：DB=DA×DA/255
    for i in range(height):
        for j in range(width):
            gray = int(grayImage[i, j]) * int(grayImage[i, j]) / 255
            result[i, j] = np.uint8(gray)
    # 显示图像
    cv2.imshow("Gray Image", grayImage)
    cv2.imshow("Result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#绘制曲线
def log_plot(c):
    x = np.arange(0, 256, 0.01)
    y = c * np.log(1 + x)
    plt.plot(x, y, 'r', linewidth=1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
    plt.title('对数变换函数')
    plt.xlabel('x-pixel')
    plt.ylabel('y-pixel')
    plt.xlim(0, 255), plt.ylim(0, 255)
    plt.show()

#对数变换
def log(c, img):
    output = c * np.log(1.0 + img)
    output = np.uint8(output + 0.5)
    return output

# 图像灰度的对数变换一般表示如公式（13-1）所示：
# D2 = c * log(1 + D1)
# 对数变换对于整体对比度偏低
# 并且灰度值偏低的图像增强效果较好
def nonlinearLog():
    # 读取原始图像
    img = cv2.imread('res/sc_scene_2.jpg')
    # 绘制对数变换曲线
    log_plot(42)
    # 图像灰度对数变换
    output = log(42, img)
    #显示图像
    cv2.imshow('Input', img)
    cv2.imshow('Output', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#绘制曲线
def gamma_plot(c,v):
    x = np.arange(0, 256, 0.01)
    y = c * (x ** v)
    plt.plot(x,y,'r',linewidth=1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
    plt.title('伽马变换函数')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([0, 255]), plt.ylim([0, 255])
    plt.show()

#伽玛变换
def gamma(img, c, v):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut)  # 像素灰度值的映射
    output_img = np.uint8(output_img + 0.5)
    return output_img

# 伽玛变换又称为指数变换或幂次变换，是另一种常用的灰度非线性变换。图
# 像灰度的伽玛变换一般表示如公式（13-2）所示：
# D2 = c * (D1 ** γ)
# 当 γ>1 时，会拉伸图像中灰度级较高的区域，压缩灰度级较低的部分
# 当 γ<1 时，会拉伸图像中灰度级较低的区域，压缩灰度级较高的部分。
# 当 γ=1 时，该灰度变换是线性的，此时通过线性方式改变原图像。
def nonlinearGamma():
    # 读取原始图像
    img = cv2.imread('res/sc-scene-1.jpg')
    # 绘制伽玛变换曲线
    gamma_plot(0.00000005, 4.0)
    # 图像灰度伽玛变换
    output = gamma(img, 0.00000005, 4.0)
    # 显示图像
    cv2.imshow('Imput', img)
    cv2.imshow('Output', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# nonlinear1()
# nonlinearLog()
nonlinearGamma()