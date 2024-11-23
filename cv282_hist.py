"""
Matplotlib 是 Python 强大的数据可视化工具，主要用于绘制各种 2D 图
形。本小节 Python 绘制直方图主要调用 matplotlib.pyplot 库中 hist()函数
实现，它会根据数据源和像素级绘制直方图。其函数主要包括五个常用的参数，
如下所示：
n, bins, patches = plt.hist(arr, bins=50, normed=1,facecolor='green', alpha=0.75)
 arr 表示需要计算直方图的一维数组
 bins 表示直方图显示的柱数，可选项，默认值为 10
 normed 表示是否将得到的直方图进行向量归一化处理，默认值为0(新版本已没有此参数)
 facecolor 表示直方图颜色
 alpha 表示透明度
 n 为返回值，表示直方图向量
 bins 为返回值，表示各个 bin 的区间范围
 patches 为返回值，表示返回每个 bin 里面包含的数据，是一个列表
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 绘制灰色图片内容的直方图
def histGray():
    # 读取图像
    src = cv2.imread('res/lena_256_gray.png')
    # 绘制直方图
    # plt.hist(src.ravel(), 256)
    plt.hist(src.ravel(), bins=256, density=1, facecolor='green', alpha=0.75)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # 显示原始图像
    cv2.imshow("src", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 绘制的彩色直方图
def histBGR():
    src = cv2.imread('res/lena_256.png')
    # 获取 BGR 三个通道的像素值
    b, g, r = cv2.split(src)
    bEx = b.ravel()
    gEx = g.ravel()
    rEx = r.ravel()
    print(src.shape, len(src.shape), bEx.shape, len(bEx.shape))
    # 绘制直方图
    plt.figure("Lena")
    # 蓝色分量
    plt.hist(bEx, bins=256, density=1, facecolor='b', edgecolor='b', alpha=0.75)
    # 绿色分量
    plt.hist(gEx, bins=256, density=1, facecolor='g', edgecolor='g', alpha=0.75)
    # 红色分量
    plt.hist(rEx, bins=256, density=1, facecolor='r', edgecolor='r', alpha=0.75)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # 显示原始图像
    cv2.imshow("src", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def histBGREx():
    src = cv2.imread('res/lena_256.png')
    # 转换为 RGB 图像
    img_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    # 获取 BGR 三个通道的像素值
    b, g, r = cv2.split(src)
    # print(r, g, b)
    plt.figure(figsize=(8, 6))
    # 设置字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    # 原始图像
    plt.subplot(221)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title("(a)原图像")

    # 绘制蓝色分量直方图
    plt.subplot(222)
    plt.hist(b.ravel(), bins=256, density=1, facecolor='b', edgecolor='b', alpha=0.75)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("(b)蓝色分量直方图")

    # 绘制绿色分量直方图
    plt.subplot(223)
    plt.hist(g.ravel(), bins=256, density=1, facecolor='g', edgecolor='g', alpha=0.75)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("(c)绿色分量直方图")

    # 绘制红色分量直方图
    plt.subplot(224)
    plt.hist(r.ravel(), bins=256, density=1, facecolor='r', edgecolor='r', alpha=0.75)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("(d)红色分量直方图")
    plt.show()

# histGray()
# histBGR()
histBGREx()