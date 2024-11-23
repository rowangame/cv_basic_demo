# -*- coding:utf-8 -*-

"""
图像缩放（image scaling）是指对数字图像的大小进行调整的过程。在
Python 中，图像缩放主要调用 resize()函数实现，函数原型如下：

result = cv2.resize(src, dsize[, result[. fx[,  fy[, interpolation]]]])

src 表示原始图像
 dsize 表示图像缩放的大小
 result 表示图像结果
 fx 表示图像 x 轴方向缩放大小的倍数
 fy 表示图像 y 轴方向缩放大小的倍数
 interpolation表示变换方法。CV_INTER_NN表示最近邻插值；
CV_INTER_LINEAR 表 示 双 线 性 插 值 （ 缺 省 使 用 ） ；
CV_INTER_AREA 表示使用像素关系重采样，当图像缩小时，
该 方 法 可 以 避 免 波 纹 出 现 ， 当 图 像 放 大 时 ， 类 似 于
CV_INTER_NN；CV_INTER_CUBIC 表示立方插值
"""

import cv2
import numpy as np

# 输出结果如图 6-5 所示，图像缩小为(100, 200, 3)像素。注意，代码中调
# 用函数 cv2.resize(src, (200,100)) 设置新图像大小 dsize 的列数为 200，
# 行数为 100。
def test1():
    src = cv2.imread('res/sc_th.jpg')
    rows, cols = src.shape[:2]
    print(rows, cols)

    #图像缩放
    result = cv2.resize(src, (200,100))
    print(result.shape)

    #显示图像
    cv2.imshow("original",src)
    cv2.imshow("result", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 获取图片“scenery.png”的元素像素值，其 rows 值为 384，cols 值
# 为 512，接着进行宽度缩小 0.6 倍、高度放大 1.2 倍的处理，运行前后对比效
# 果如图 6-6 所示。
def test2():
    src = cv2.imread('res/sc_th.jpg')
    rows, cols = src.shape[:2]
    print(rows, cols)

    ##图像缩放 dsize(列,行)
    result = cv2.resize(src, (int(cols * 0.6), int(rows * 1.2)))
    print(result.shape)

    # 显示图像
    cv2.imshow("src", src)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 最后讲解调用(fx,fy)参数设置缩放倍数的方法，对原始图像进行放大或缩小
# 操作。下面代码是 fx 和 fy 方向缩小至原始图像 0.3 倍的操作。
def test3():
    src = cv2.imread('res/sc_th.jpg')
    rows, cols = src.shape[:2]
    print(rows, cols)

    result = cv2.resize(src, None, fx=0.3,fy=0.3)
    # 显示图像
    cv2.imshow("src", src)
    cv2.imshow("result", result)

    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test2()