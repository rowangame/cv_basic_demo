"""
图像形态学处理之开运算、闭运算和梯度运算
1.图像开运算
开运算一般能平滑图像的轮廓，削弱狭窄部分，去掉较细的突出。闭运算也
是平滑图像的轮廓，与开运算相反，它一般熔合窄的缺口和细长的弯口，去掉小
洞，填补轮廓上的缝隙。图像开运算是图像依次经过腐蚀、膨胀处理的过程，图
像被腐蚀后将去除噪声，但同时也压缩了图像，接着对腐蚀过的图像进行膨胀处
理，可以在保留原有图像的基础上去除噪声

2.图像闭运算
图像闭运算是图像依次经过膨胀、腐蚀处理的过程，先膨胀后腐蚀有助于过
滤前景物体内部的小孔或物体上的小黑点

3.图像梯度运算
图像梯度运算是图像膨胀处理减去图像腐蚀处理后的结果，从而得到图像的
轮廓
"""

import cv2
import numpy as np

# 图像开运算
def morphologyExOpen():
    # 读取图片
    src = cv2.imread('res/ns_pic_x.jpg', cv2.IMREAD_UNCHANGED)
    # 设置卷积核
    kernel = np.ones((5, 5), np.uint8)
    # 图像开运算
    result = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
    # 显示图像
    cv2.imshow("src", src)
    cv2.imshow("result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 2.图像闭运算
# 图像闭运算是图像依次经过膨胀、腐蚀处理的过程，先膨胀后腐蚀有助于过
# 滤前景物体内部的小孔或物体上的小黑点
def morphologyExClose():
    # 读取图片
    src = cv2.imread('res/ns_pic_x.jpg', cv2.IMREAD_UNCHANGED)
    # 设置卷积核
    kernel = np.ones((5, 5), np.uint8)
    # 图像闭运算
    result = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
    # 显示图像
    cv2.imshow("src", src)
    cv2.imshow("result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 3.图像梯度运算
# 图像梯度运算是图像膨胀处理减去图像腐蚀处理后的结果，从而得到图像的轮廓
def morphologyExGradient():
    # 读取图片
    src = cv2.imread('res/ns_pic_x.jpg', cv2.IMREAD_UNCHANGED)
    # 设置卷积核
    kernel = np.ones((5, 5), np.uint8)
    # 图像闭运算
    result = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)
    # 显示图像
    cv2.imshow("src", src)
    cv2.imshow("result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 1.图像顶帽运算
# 图像顶帽运算（top-hat transformation）又称为图像礼帽运算，它是用
# 原始图像减去图像开运算后的结果，常用于解决由于光照不均匀图像分割出错的问题
def morphologyExTopHat():
    # 读取图片
    src = cv2.imread('res/bar_rice_1.jpg', cv2.IMREAD_UNCHANGED)
    # 设置卷积核
    kernel = np.ones((10, 10), np.uint8)
    # 图像顶帽运算
    result = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel)
    # 显示图像
    cv2.imshow("src", src)
    cv2.imshow("result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 为什么图像顶帽运算会消除光照不均匀的效果呢？通常可以利用灰度三维
# 图来进行解释该算法。灰度三维图主要调用 Axes3D 包实现，对原图绘制灰度
# 三维图的代码如下：
def testSurfacePlot():
    import numpy as np
    import cv2 as cv
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator,FormatStrFormatter

    # 读取图像
    img = cv.imread('res/bar_rice_1.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 图像顶帽运算
    boUseTopHat = True
    if boUseTopHat:
        # 通过图像白帽运算后的图像灰度三维图如图
        # 所示，对应的灰度更集中于10至100区间，
        # 由此证明了不均匀的背景被大致消除了，有利于后续的阈值分割或图像分割
        # 设置卷积核
        kernel = np.ones((10, 10), np.uint8)
        # 图像顶帽运算
        result = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        imgd = np.array(result)  # image 类转 numpy
    else:
        imgd = np.array(img)

    # 准备数据
    sp = img.shape
    h = int(sp[0])  # 图像高度(rows)
    w = int(sp[1])  # 图像宽度(colums) of image
    # 绘图初始处理
    fig = plt.figure(figsize=(14, 10))
    ax = fig.gca(projection="3d")
    oriX = np.arange(0, w, 1)
    oriY = np.arange(0, h, 1)
    x, y = np.meshgrid(oriX, oriY)
    z = imgd
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)
    # 自定义 z 轴
    ax.set_zlim(-10, 255)
    ax.zaxis.set_major_locator(LinearLocator(10))  # 设置 z 轴网格线的疏密
    # 将 z 的 value 字符串转为 float 并保留 2 位小数
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # 设置坐标轴的 label 和标题
    ax.set_xlabel('x', size=15)
    ax.set_ylabel('y', size=15)
    ax.set_zlabel('z', size=15)
    ax.set_title("surface plot", weight='bold', size=20)
    # 添加右侧的色卡条
    fig.colorbar(surf, shrink=0.6, aspect=8)
    # 保存当前场景
    # plt.savefig('./out/morphologyEx-2.png')
    plt.show()


# 图像底帽运算是用一个结构元通过闭运算从一幅图像中删除物体，常用于校
# 正不均匀光照的影响
def morphologyExBlackHat():
    # 读取图片
    src = cv2.imread('res/bar_rice_1.jpg', cv2.IMREAD_UNCHANGED)
    # 设置卷积核
    kernel = np.ones((10, 10), np.uint8)
    # 图像顶帽运算
    result = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel)
    # 显示图像
    cv2.imshow("src", src)
    cv2.imshow("result", result)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# morphologyExOpen()
# morphologyExClose()
# morphologyExGradient()
# morphologyExTopHat()
testSurfacePlot()
# morphologyExBlackHat()