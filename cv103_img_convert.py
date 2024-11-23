
"""
在日常生活中，我们看到的大多数彩色图像都是 RGB 类型，但是在图像处
理过程中，常常需要用到灰度图像、二值图像、HSV、HSI 等颜色。图像类型
转换是指将一种类型转换为另一种类型，比如彩色图像转换为灰度图像、BGR
图像转换为 RGB 图像。OpenCV 提供了 200 多种不同类型之间的转换，其
中最常用的包括 3 类，如下：
 cv2.COLOR_BGR2GRAY
 cv2.COLOR_BGR2RGB
 cv2.COLOR_GRAY2BGR
OpenCV 提供了 cvtColor()函数实现这些功能。其函数原型如下所示：
 dst = cv2.cvtColor(src, code[, dst[, dstCn]])
 src 表示输入图像，需要进行颜色空间变换的原图像
 dst 表示输出图像，其大小和深度与 src 一致
 code 表示转换的代码或标识
 dstCn 表示目标图像通道数，其值为 0 时，则有 src 和 code 决定
该函数的作用是将一个图像从一个颜色空间转换到另一个颜色空间，其中，
RGB 是指 Red、Green 和 Blue，一副图像由这三个通道（channel）构成；
Gray 表示只有灰度值一个通道；HSV 包含 Hue（色调）、Saturation（饱
和度）和 Value（亮度）三个通道。在 OpenCV 中，常见的颜色空间转换标识
包 括 CV_BGR2BGRA 、 CV_RGB2GRAY 、 CV_GRAY2RGB 、
CV_BGR2HSV、CV_BGR2XYZ、CV_BGR2HLS
"""

import cv2
import matplotlib.pyplot as plt

#读取原始图像
img_BGR = cv2.imread("res/lena_256.png")

#BGR 转换为 RGB
img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

#灰度化处理
img_GRAY = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)

#BGR 转 HSV
img_HSV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)

#BGR 转 YCrCb
img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)

#BGR 转 HLS
img_HLS = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HLS)

#BGR 转 XYZ
img_XYZ = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2XYZ)

#BGR 转 LAB
img_LAB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2LAB)

#BGR 转 YUV
img_YUV = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YUV)

#调用 matplotlib 显示处理结果
titles = ['BGR', 'RGB', 'GRAY', 'HSV', 'YCrCb', 'HLS', 'XYZ', 'LAB', 'YUV']
images = [img_BGR, img_RGB, img_GRAY, img_HSV, img_YCrCb, img_HLS, img_XYZ, img_LAB, img_YUV]
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()