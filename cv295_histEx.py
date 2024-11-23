
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1.灰度增强直方图对比
# D2 = D1 + 50
def histByLinear1():
    # 读取图像
    img = cv2.imread('res/lena_256_gray.png')
    # 图像灰度转换
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = grayImage.shape[0]
    width = grayImage.shape[1]
    result = np.zeros((height, width), np.uint8)
    # 图像灰度上移变换
    for i in range(height):
        for j in range(width):
            if (int(grayImage[i, j] + 50) > 255):
                gray = 255
            else:
                gray = int(grayImage[i, j] + 50)
            result[i, j] = np.uint8(gray)
    # 计算原图的直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    # 计算灰度变换的直方图
    hist_res = cv2.calcHist([result], [0], None, [256], [0, 255])
    # 原始图像
    plt.figure(figsize=(8, 6))
    plt.subplot(221), plt.imshow(img, 'gray'), plt.title("(a)"),
    plt.axis('off')

    # 绘制掩膜
    plt.subplot(222), plt.plot(hist), plt.title("(b)"), plt.xlabel("x"),
    plt.ylabel("y")

    # 绘制掩膜设置后的图像
    plt.subplot(223), plt.imshow(result, 'gray'), plt.title("(c)"),
    plt.axis('off')

    # 绘制直方图
    plt.subplot(224), plt.plot(hist_res), plt.title("(d)"),
    plt.xlabel("x"), plt.ylabel("y")
    plt.show()

# 2.灰度减弱直方图对比
# D2 = D1 * 0.8
def histByLinear2():
    # 读取图像
    img = cv2.imread('res/lena_256_gray.png')
    # 图像灰度转换
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取图像高度和宽度
    height = grayImage.shape[0]
    width = grayImage.shape[1]
    result = np.zeros((height, width), np.uint8)
    # 图像对比度减弱变换 DB=DA×0.8
    for i in range(height):
        for j in range(width):
            gray = int(grayImage[i, j] * 0.8)
            result[i, j] = np.uint8(gray)

    #计算原图的直方图
    hist = cv2.calcHist([img], [0], None, [256], [0,255])
    #计算灰度变换的直方图
    hist_res = cv2.calcHist([result], [0], None, [256], [0,255])
    # 原始图像
    plt.figure(figsize=(8, 6))
    plt.subplot(221), plt.imshow(img, 'gray'), plt.title("(a)"),
    plt.axis('off')
    # 绘制原图直方图
    plt.subplot(222), plt.plot(hist), plt.title("(b)"), plt.xlabel("x"),
    plt.ylabel("y")

    #绘制变化后的直方图
    plt.subplot(223), plt.imshow(result, 'gray'), plt.title("(c)"),
    plt.axis('off')
    # 绘制直方图
    plt.subplot(224), plt.plot(hist_res), plt.title("(d)"),
    plt.xlabel("x"), plt.ylabel("y")
    plt.show()

# 3.图像反色直方图对比
# 该算法将图像的颜色反色，对原图像的像素值进行反转，即黑色变为白色，
# 白色变为黑色，使用的表达式为：
# D2 = 255 - D1
def histByInvColor():
    img = cv2.imread('res/lena_256.png')
    # 图像灰度转换
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取图像高度和宽度
    height = grayImage.shape[0]
    width = grayImage.shape[1]
    result = np.zeros((height, width), np.uint8)
    # 图像灰度反色变换 DB=255-DA
    for i in range(height):
        for j in range(width):
            gray = 255 - grayImage[i,j]
            result[i,j] = np.uint8(gray)
    # 计算原图的直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    # 计算灰度变换的直方图
    hist_res = cv2.calcHist([result], [0], None, [256], [0, 255])

    # 原始图像
    plt.figure(figsize=(8, 6))
    plt.subplot(221), plt.imshow(img, 'gray'), plt.title("(a)"),
    plt.axis('off')
    # 绘制原图的直方图
    plt.subplot(222), plt.plot(hist), plt.title("(b)"), plt.xlabel("x"),
    plt.ylabel("y")
    # 绘制变换后的直方图
    plt.subplot(223), plt.imshow(result, 'gray'), plt.title("(c)"),
    plt.axis('off')
    # 绘制直方图
    plt.subplot(224), plt.plot(hist_res), plt.title("(d)"),
    plt.xlabel("x"), plt.ylabel("y")
    plt.show()

# 4.图像对数变换直方图对比
# 该算法将增加低灰度区域的对比度，从而增强暗部的细节，使用的表达式为：
# D2 = c * log(1 + D1)
def histByLog():
    img = cv2.imread('res/lena_256_gray.png')
    # 图像灰度转换
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 获取图像高度和宽度
    height = grayImage.shape[0]
    width = grayImage.shape[1]
    result = np.zeros((height, width), np.uint8)
    # 图像灰度对数变换
    for i in range(height):
        for j in range(width):
            gray = 42 * np.log(1.0 + grayImage[i, j])
            result[i, j] = np.uint8(gray)
    # 计算原图的直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    # 计算灰度变换的直方图
    hist_res = cv2.calcHist([result], [0], None, [256], [0, 255])
    # 原始图像
    plt.figure(figsize=(8, 6))
    plt.subplot(221), plt.imshow(img, 'gray'), plt.title("(a)"),
    plt.axis('off')
    # 绘制原始图像直方图
    plt.subplot(222), plt.plot(hist), plt.title("(b)"), plt.xlabel("x"),
    plt.ylabel("y")
    # 灰度变换后的图像
    plt.subplot(223), plt.imshow(result, 'gray'), plt.title("(c)"),
    plt.axis('off')
    # 灰度变换图像的直方图
    plt.subplot(224), plt.plot(hist_res), plt.title("(d)"),
    plt.xlabel("x"), plt.ylabel("y")
    plt.show()

# 5.图像阈值化处理直方图对比
def histByThreshold():
    img = cv2.imread('res/lena_256_gray.png')
    # 图像灰度转换
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二进制阈值化处理
    r, result = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    # 计算原图的直方图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # 计算阈值化处理的直方图
    hist_res = cv2.calcHist([result], [0], None, [256], [0, 256])
    # 原始图像
    plt.figure(figsize=(8, 6))
    plt.subplot(221), plt.imshow(img, 'gray'), plt.title("(a)"),
    plt.axis('off')
    # 绘制原始图像直方图
    plt.subplot(222), plt.plot(hist), plt.title("(b)"), plt.xlabel("x"),
    plt.ylabel("y")
    # 阈值化处理后的图像
    plt.subplot(223), plt.imshow(result, 'gray'), plt.title("(c)"),
    plt.axis('off')
    # 阈值化处理图像的直方图
    plt.subplot(224), plt.plot(hist_res), plt.title("(d)"),
    plt.xlabel("x"), plt.ylabel("y")
    plt.show()

# histByLinear1()
# histByLinear2()
# histByInvColor()
# histByLog()
histByThreshold()