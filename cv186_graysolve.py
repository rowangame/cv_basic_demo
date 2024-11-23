# 该方法的灰度值等于彩色图像 R、G、B 三个分量中的最大值，公式如下：
# gray(i, j)  max(R(i, j),G(i, j),B(i, j))

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 其方法灰度化处理后的灰度图亮度很高
def grayByMaxValue():
    img = cv2.imread("res/lena_256.png")
    # 获取图像高度和宽度
    height = img.shape[0]
    width = img.shape[1]
    # 创建一幅图像
    grayimg = np.zeros((height, width, 3), np.uint8)
    # 图像最大值灰度处理
    for i in range(height):
        for j in range(width):
            # 获取图像 R G B 最大值
            gray = max(img[i, j][0], img[i, j][1], img[i, j][2])
            # 灰度图像素赋值 gray=max(R,G,B)
            grayimg[i, j] = np.uint8(gray)
    # 显示图像
    cv2.imshow("src", img)
    cv2.imshow("gray", grayimg)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 该方法的灰度值等于彩色图像 R、G、B 三个分量灰度值的求和平均值
def grayByAvgValue():
    img = cv2.imread("res/lena_256.png")
    height,width = img.shape[0], img.shape[1]
    grayimg = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            # 这里uint8类型累加会溢出，会丢失精度,需要转化为整型
            gray = (int(0) + img[i,j][0] + img[i,j][1] + img[i,j][2]) / 3
            grayimg[i,j] = np.uint8(gray)
    cv2.imshow("src", img)
    cv2.imshow("gray", grayimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grayByRightAvgValue():
    img = cv2.imread("res/lena_256.png")
    height,width = img.shape[0], img.shape[1]
    grayimg = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            # 灰度加权平均法 (gray=0.299∗R+0.578∗G+0.114∗B)
            gray = 0.11 * img[i, j][0] + 0.59 * img[i, j][1] + 0.30 * img[i, j][2]
            grayimg[i, j] = np.uint8(gray)
    cv2.imshow("src", img)
    cv2.imshow("gray", grayimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test1():
    width = 8
    height = 4
    channel = 3
    img = np.zeros((height, width, channel), np.uint8)
    print("-----------img--------------")
    print(img)
    print("-----------img[0]--------------")
    img[0:height,0:width,0:channel] = 125
    print(img[0].shape)
    print(img[0])
    print("-----------img[0,0]--------------")
    print(img[0,0].shape)
    print(img[0,0])
    print("-----------img[0,0,0]--------------")
    print(img[0,0,0].shape)
    print(img[0,0,0])

def test2():
    width = 8
    height = 4
    channel = 3
    img = np.zeros((height, width, channel), np.uint8)
    for tmpH in range(height):
        for tmpW in range(width):
            img[tmpH, tmpW,0:channel] = [tmpH, tmpH, tmpH]
    print("-------------all---------------")
    print(img)
    print("-------------img[1]---------------")
    print(img[1])

if __name__ == "__main__":
    # grayByMaxValue()
    test1()
    # test2()
    # grayByAvgValue()