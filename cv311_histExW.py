import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 1.图像掩膜直方图
# 如果要统计图像的某一部分直方图，就需要使用掩码（蒙板）来进行计算。
# 假设将要统计的部分设置为白色，其余部分设置为黑色，然后使用该掩膜进行直
# 方图绘制
def histByMask():
    img = cv2.imread('res/lena_256_gray.png')
    # 转换为 RGB 图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 设置掩膜
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[80:150, 80:150] = 255
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # 图像直方图计算
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])  # 通道[0] - 灰度图
    # 图像直方图计算(含掩膜)
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
    plt.figure(figsize=(8, 6))

    # 设置字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # 原始图像
    plt.subplot(221)
    plt.imshow(img_rgb, 'gray')
    plt.axis('off')
    plt.title("(a)原始图像")

    # 绘制掩膜
    plt.subplot(222)
    plt.imshow(mask, 'gray')
    plt.axis('off')
    plt.title("(b)掩膜")

    # 绘制掩膜设置后的图像
    plt.subplot(223)
    plt.imshow(masked_img, 'gray')
    plt.axis('off')
    plt.title("(c)图像掩膜处理")

    # 绘制直方图
    plt.subplot(224)
    plt.plot(hist_full, 'b')
    plt.plot(hist_mask, 'r')
    plt.title("(d)直方图曲线")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def histByHSV():
    img = cv2.imread('res/lena_256.png')
    # 转换为 RGB 图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 图像 HSV 转换
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 计算 H-S 直方图
    hist = cv2.calcHist(img_hsv, [0, 1], None, [180, 256], [0, 180, 0, 256])
    # histH = cv2.calcHist(img_hsv, [0], None, [180], [0, 180])
    # histS = cv2.calcHist(img_hsv, [1], None, [256], [0, 256])

    # 原始图像
    plt.figure(figsize=(8,6))
    plt.subplot(121)
    plt.imshow(img_rgb, "gray")
    plt.title("(a)")
    plt.axis("off")

    # 绘制 H-S 直方图
    plt.subplot(122)
    plt.imshow(hist, interpolation='nearest')
    # plt.plot(hist, 'b')
    # plt.plot(histH, 'b')
    # plt.plot(histS, 'r')
    plt.title("(b)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

histByMask()
# histByHSV()