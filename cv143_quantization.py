# -*- coding: utf-8 -*-

"""
量化（Quantization）旨在将图像像素点对应亮度的连续变化区间转换为
单个特定值的过程，即将原始灰度图像的空间坐标幅度值离散化。量化等级越多，
图像层次越丰富，灰度分辨率越高，图像的质量也越好；量化等级越少，图像层
次欠丰富，灰度分辨率越低，会出现图像轮廓分层的现象，降低了图像的质量。
图 8-1 是将图像的连续灰度值转换为 0 至 255 的灰度级的过程
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
如果量化等级为 2，则将使用两种灰度级表示原始图片的像素（0~255），
灰度值小于 128 的取 0，大于等于 128 的取 128；如果量化等级为 4，则将使
用四种灰度级表示原始图片的像素，新图像将分层为四种颜色，0~64区间取0，
64~128 区间取 64，128~192 区间取 128，192~255 区间取 192，依次类
推。
图 8-2 是对比不同量化等级的“Lena”图。其中（a）的量化等级为 256，
（b）的量化等级为 64，（c）的量化等级为 16，（d）的量化等级为 8，（e）
的量化等级为 4，（f）的量化等级为 2。
"""

def scaleImage(img: np.ndarray, scaleX, scaleY: float):
    result = cv2.resize(img, None, fx=scaleX, fy=scaleY)
    return result

def grayImage(img: np.ndarray):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 保存图片
def saveImg(img: np.ndarray, filename: str):
    cv2.imwrite(filename, img)

def solveImg():
    img = cv2.imread("res/lena_512.png")
    img1 = scaleImage(img, 0.5, 0.5)
    img2 = scaleImage(img, 0.25, 0.25)
    saveImg(img1,"./out/lena-1-hd.png")
    saveImg(img2, "./out/lena-2-hd.png")
    saveImg(img, "./out/lena.png")
    img1gray = grayImage(img1)
    img2gray = grayImage(img2)
    imggray = grayImage(img)
    saveImg(img1gray,"./out/lena-1-hd-gray.png")
    saveImg(img2gray, "./out/lena-2-hd-gray.png")
    saveImg(imggray, "./out/lena-gray.png")

# 图像量化的实现过程是建立一张临时图片，接着循环遍历原始图像中所有像
# 素点，判断每个像素点应该属于的量化等级，最后将临时图像显示。下面的代码
# 将灰度图像转换为两种量化等级
def test1():
    img = cv2.imread('res/lena_256_gray.png')
    print(img.shape)

    # 获取图像高度和宽度
    height = img.shape[0]
    width = img.shape[1]
    # 创建一幅图像
    new_img = np.zeros((height, width, 3), np.uint8)
    # 图像量化操作 量化等级为 2
    for i in range(height):
        for j in range(width):
            for k in range(3):  # 对应 BGR 三分量
                if img[i, j][k] < 128:
                    gray = 0
                else:
                    gray = 128
                new_img[i, j][k] = np.uint8(gray)
    # 显示图像
    cv2.imshow("src", img)
    cv2.imshow("", new_img)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 下面的代码分别比较了量化等级为 2、4、8 的量化处理效果
def test2():
    # 读取原始图像
    img = cv2.imread('res/lena_256_gray.png')
    # 获取图像高度和宽度
    height = img.shape[0]
    width = img.shape[1]
    # 创建一幅图像
    new_img1 = np.zeros((height, width, 3), np.uint8)
    new_img2 = np.zeros((height, width, 3), np.uint8)
    new_img3 = np.zeros((height, width, 3), np.uint8)
    # 图像量化等级为 2 的量化处理
    for i in range(height):
        for j in range(width):
            for k in range(3):  # 对应 BGR 三分量
                if img[i, j][k] < 128:
                    gray = 0
                else:
                    gray = 128
                new_img1[i, j][k] = np.uint8(gray)
    # 图像量化等级为 4 的量化处理
    for i in range(height):
        for j in range(width):
            for k in range(3):  # 对应 BGR 三分量
                if img[i, j][k] < 64:
                    gray = 0
                elif img[i, j][k] < 128:
                    gray = 64
                elif img[i, j][k] < 192:
                    gray = 128
                else:
                    gray = 192
                new_img2[i, j][k] = np.uint8(gray)

    # 图像量化等级为 8 的量化处理
    for i in range(height):
        for j in range(width):
            for k in range(3):  # 对应 BGR 三分量
                if img[i, j][k] < 32:
                    gray = 0
                elif img[i, j][k] < 64:
                    gray = 32
                elif img[i, j][k] < 96:
                    gray = 64
                elif img[i, j][k] < 128:
                    gray = 96
                elif img[i, j][k] < 160:
                    gray = 128
                elif img[i, j][k] < 192:
                    gray = 160
                elif img[i, j][k] < 224:
                    gray = 192
                else:
                    gray = 224
                new_img3[i, j][k] = np.uint8(gray)

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图像
    titles = ['(a) 原始图像', '(b) 量化-L2', '(c) 量化-L4', '(d) 量化L8']
    images = [img, new_img1, new_img2, new_img3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray'),
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    # test1()
    test2()