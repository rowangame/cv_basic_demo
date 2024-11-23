"""
局部直方图均衡化
 OpenCV 中 equalizeHist()函数实现直方图均衡化处理，
该方法简单高效，但其实它是一种全局意义上的均衡化处理，很多时候这种操作
不是很好，会把某些不该调整的部分给均衡处理了。同时，图像中不同的区域灰
度分布相差甚远，对它们使用同一种变换常常产生不理想的效果，实际应用中，
常常需要增强图像的某些局部区域的细节。
为了解决这类问题，Pizer 等提出了局部直方图均衡化的方法（AHE），但
AHE 方法仅仅考虑了局部区域的像素，忽略了图像其他区域的像素，且对于图
像中相似区域具有过度放大噪声的缺点。为此 K. Zuiderveld 等人提出了对比
度受限 CLAHE 的图像增强方法，通过限制局部直方图的高度来限制局部对比
度的增强幅度，从而限制噪声的放大及局部对比度的过增强，该方法常用于图像
增强，也可以被用来进行图像去雾操作
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 在 OpenCV 中，调用函数 createCLAHE()实现对比度受限的局部直方
# 图均衡化。它将整个图像分成许多小块（比如按 10×10 作为一个小块），那么
# 对每个小块进行均衡化。这种方法主要对于图像直方图不是那么单一的（比如存
# 在多峰情况）图像比较实用。其函数原型如下所示：
#  retval = createCLAHE([, clipLimit[, tileGridSize]])
#  clipLimit 参数表示对比度的大小
#  tileGridSize 参数表示每次处理块的大小
def createCLAHE():
    # 读取图片
    img = cv2.imread('res/lena_256.png')
    # 灰度转换
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #局部直方图均衡化处理
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(10,10))
    #将灰度图像和局部直方图相关联, 把直方图均衡化应用到灰度图
    result = clahe.apply(gray)

    #显示图像
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

createCLAHE()