
import cv2
import numpy as np

"""
图像属性
图像中最常见的三个属性进行介绍，它们分别是图像形状（shape）、像素大小
（size）和图像类型（dtype）。
"""
def testCommonProperty():
    # 读取图片
    img = cv2.imread("res/lena_256.png")
    # 获取图像形状
    print("shape:", img.shape)

    # 通过 size 关键字获取图像的像素数目，其中灰度图像返回行数×列数，彩
    # 色图像返回行数×列数×通道数
    print("size:", img.size)

    # 通过 dtype 关键字获取图像的数据类型，通常返回 uint8
    print("dtype:", img.dtype)

    # 显示图像
    cv2.imshow("Demo", img)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
图像通道分离及合并
OpenCV 通过 split()函数和 merge()函数实现对图像通道的处理，包括
通道分离和通道合并。
（1）split()函数
OpenCV 读取的彩色图像由蓝色（B）、绿色（G）、红色（R）三原色组
成，每一种颜色可以认为是一个通道分量
"""
def testSplit():
    # 读取图片
    img = cv2.imread("res/lena_256.png")
    # 拆分通道
    b, g, r = cv2.split(img)
    # 显示原始图像
    cv2.imshow("B", b)
    cv2.imshow("G", g)
    cv2.imshow("R", r)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def testMerge():
    # 读取图片
    img = cv2.imread("res/lena_256.png")
    # 拆分通道
    b, g, r = cv2.split(img)
    # 合并通道
    m = cv2.merge([b, g, r])
    cv2.imshow("Merge", m)

    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
可以调用该函数提取图像的不同颜色，比如提取 B 颜色通道，G、B
通道设置为 0
"""
def testMergeEx():
    # 读取图片
    img = cv2.imread("res/lena_256.png")
    rows, cols, chn = img.shape
    # 拆分通道
    b = cv2.split(img)[0]
    # 设置 g、r 通道为 0
    g = np.zeros((rows, cols), dtype=img.dtype)
    r = np.zeros((rows, cols), dtype=img.dtype)
    # 合并通道
    m = cv2.merge([b, g, r])
    cv2.imshow("Merge", m)
    #等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# testCommonProperty()
# testSplit()
# testMerge()
testMergeEx()