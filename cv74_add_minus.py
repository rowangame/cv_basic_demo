import cv2
import numpy as np

"""
1.图像加法运
图像加法运算主要有两种方法。第一种是调用 Numpy 库实现，目标图像像
素为两张图像的像素之和；第二种是通过 OpenCV 调用 add()函数实现。第二
种方法的函数原型如下：
dst = add(src1, src2[, dst[, mask[, dtype]]])
 src1 表示第一张图像的像素矩阵
 src2 表示第二张图像的像素矩阵
 dst 表示输出的图像，必须和输入图像具有相同的大小和通道数
 mask 表示可选操作掩码（8 位单通道数组），用于指定要更改的输出
数组的元素。
 dtype 表示输出数组的可选深度

注意，当两幅图像的像素值相加结果小于等于 255 时，则输出图像直接赋
值该结果，如 120+48 赋值为 168；如果相加值大于 255，则输出图像的像素
结果设置为 255，如(255+64) 赋值为 255。下面的代码实现了图像加法运算。
"""
def imgAdd():
    # 读取图片
    img = cv2.imread("res/lena_256.png")
    # 图像各像素加 100
    m = np.ones(img.shape, dtype="uint8") * 100
    # OpenCV 加法运算
    result = cv2.add(img, m)
    # 显示图像
    cv2.imshow("original", img)
    cv2.imshow("result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
dst = subtract(src1, src2[, dst[, mask[, dtype]]])
 src1 表示第一张图像的像素矩阵
 src2 表示第二张图像的像素矩阵
 dst 表示输出的图像，必须和输入图像具有相同的大小和通道数
 mask 表示可选操作掩码（8 位单通道数组），用于指定要更改的输
出数组的元素。
"""
def imgMinuse():
    # 读取图片
    img = cv2.imread("res/lena_256.png")
    # 图像各像素减 50
    m = np.ones(img.shape, dtype="uint8") * 90
    # OpenCV 减法运算
    result = cv2.subtract(img, m)
    # 显示图像
    cv2.imshow("original", img)
    cv2.imshow("result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

imgAdd()
# imgMinuse()