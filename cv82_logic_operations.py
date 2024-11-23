
import cv2
import numpy as np

"""
图像与运算
与运算是计算机中一种基本的逻辑运算方式，符号表示为“&”，其运算规
则为：
 0&0=0
 0&1=0
 1&0=0
 1&1=1
图像的与运算是指两张图像（灰度图像或彩色图像均可）的每个像素值进行
二进制“与”操作，实现图像裁剪。

dst = bitwise_and(src1, src2[, dst[, mask]])
 src1 表示第一张图像的像素矩阵
 src2 表示第二张图像的像素矩阵
 dst 表示输出的图像，必须和输入图像具有相同的大小和通道数
 mask 表示可选操作掩码（8 位单通道数组），用于指定要更改的输
出数组的元素。
"""
def op_and():
    # 读取图片
    img = cv2.imread("res/lena_256.png", cv2.IMREAD_GRAYSCALE)
    # 获取图像宽和高
    rows, cols = img.shape[:2]
    print(rows, cols)

    # 画圆形
    circle = np.zeros((rows, cols), dtype="uint8")
    cv2.circle(circle, (int(rows / 2), int(cols / 2)), 100, 255, -1)
    print(circle.shape)
    print(img.size, circle.size)

    # OpenCV 图像与运算
    result = cv2.bitwise_and(img, circle)
    # 显示图像
    cv2.imshow("original", img)
    cv2.imshow("circle", circle)
    cv2.imshow("result", result)

    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
逻辑或运算是指如果一个操作数或多个操作数为 true，则逻辑或运算符返
回布尔值 true；只有全部操作数为 false，结果才是 false。图像的或运算是
指两张图像（灰度图像或彩色图像均可）的每个像素值进行二进制“或”操作，
实现图像裁剪。其函数原型如下所示：
dst = bitwise_or(src1, src2[, dst[, mask]])
 src1 表示第一张图像的像素矩阵
 src2 表示第二张图像的像素矩阵
 dst 表示输出的图像，必须和输入图像具有相同的大小和通道数
 mask 表示可选操作掩码（8 位单通道数组），用于指定要更改的输
出数组的元素。
"""
def op_or():
    # 读取图片
    img = cv2.imread("res/lena_256.png", cv2.IMREAD_GRAYSCALE)
    # 获取图像宽和高
    rows, cols = img.shape[:2]
    # 画圆形
    circle = np.zeros((rows, cols), dtype="uint8")
    cv2.circle(circle, (int(rows / 2), int(cols / 2)), 100, 255, -1)
    # OpenCV 图像或运算
    result = cv2.bitwise_or(img, circle)
    # 显示图像
    cv2.imshow("original", img)
    cv2.imshow("circle", circle)
    cv2.imshow("result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
图像非运算就是图像的像素反色处理，它将原始图像的黑色像素点转换为白
色像素点，白色像素点则转换为黑色像素点，其函数原型如下：
dst = bitwise_not(src1, src2[, dst[, mask]])
 src1 表示第一张图像的像素矩阵
 src2 表示第二张图像的像素矩阵
 dst 表示输出的图像，必须和输入图像具有相同的大小和通道数
 mask 表示可选操作掩码（8 位单通道数组），用于指定要更改的输
出数组的元素。
注:通过非运算后的结果图像与原图像值加起来都是255
"""
def op_not():
    # 读取图片
    img = cv2.imread("res/sc_night.jpg", cv2.IMREAD_GRAYSCALE)
    # OpenCV 图像非运算
    result = cv2.bitwise_not(img)
    # add(通过非运算后的结果图像与原图像值加起来都是255)
    imgAdd = cv2.add(img, result)
    # 显示图像
    cv2.imshow("original", img)
    cv2.imshow("result", result)
    cv2.imshow("add", imgAdd)

    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
逻辑异或运算（xor）是一个数学运算符，数学符号为“⊕”，计算机符号
为“xor”，其运算法则为：如果 a、b 两个值不相同，则异或结果为 1；如果
a、b 两个值相同，异或结果为 0。
图像的异或运算是指两张图像（灰度图像或彩色图像均可）的每个像素值进
行二进制“异或”操作，实现图像裁剪。其函数原型如下所示：
dst = bitwise_xor(src1, src2[, dst[, mask]])
 src1 表示第一张图像的像素矩阵
 src2 表示第二张图像的像素矩阵
 dst 表示输出的图像，必须和输入图像具有相同的大小和通道数
 mask 表示可选操作掩码（8 位单通道数组），用于指定要更改的
输出数组的元素。
"""
def op_xor():
    # 读取图片
    img = cv2.imread("res/lena_256.png", cv2.IMREAD_GRAYSCALE)
    # 获取图像宽和高
    rows, cols = img.shape[:2]
    # 画圆形
    circle = np.zeros((rows, cols), dtype="uint8")
    cv2.circle(circle, (int(rows / 2), int(cols / 2)), 100, 255, -1)
    # OpenCV 图像异或运算
    result = cv2.bitwise_xor(img, circle)
    # 显示图像
    cv2.imshow("original", img)
    cv2.imshow("circle", circle)
    cv2.imshow("result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# op_and()
# op_or()
# op_not()
op_xor()