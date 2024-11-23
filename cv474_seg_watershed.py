"""
图像分水岭算法（Watershed Algorithm）是将图像的边缘轮廓转换为
“山脉”，将均匀区域转换为“山谷”，从而提升分割效果的算法[3]。分水岭算
法是基于拓扑理论的数学形态学的分割方法，灰度图像根据灰度值把像素之间的
关系看成山峰和山谷的关系，高亮度（灰度值高）的地方是山峰，低亮度（灰度
值低）的地方是山谷。接着给每个孤立的山谷（局部最小值）不同颜色的水
（Label），当水涨起来，根据周围的山峰（梯度），不同的山谷也就是不同颜
色的像素点开始合并，为了避免这个现象，可以在水要合并的地方建立障碍，直
到所有山峰都被淹没。所创建的障碍就是分割结果，这个就是分水岭的原理

分水岭算法的计算过程是一个迭代标注过程，主要包括排序和淹没两个步骤。
由于图像会存在噪声或缺失等问题，该方法会造成分割过度。OpenCV 提供了
watershed()函数实现图像分水岭算法，并且能够指定需要合并的点，其函数
原型如下所示：
markers = watershed(image, markers)
 image 表示输入图像，需为 8 位三通道的彩色图像
 markers 表示用于存储函数调用之后的运算结果，输入/输出 32
位单通道图像的标记结构，输出结果需和输入图像的尺寸和类型一致。
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# step1: 阈值化处理
def threshold():
    # 读取原始图像
    img = cv2.imread('res/bar_coin_3.png')
    # 图像灰度化处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图像阈值化处理
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    # 显示图像
    cv2.imshow('src', img)
    cv2.imshow('gray', gray)
    cv2.imshow('res', thresh)
    cv2.waitKey()
    cv2.destroyAllWindows()

# step2: 形态学处理(过滤掉小的白色噪声)
def morphologyEx():
    # 读取原始图像
    img = cv2.imread('res/bar_coin_3.png')
    # 图像灰度化处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图像阈值化处理
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 图像开运算消除噪声
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # 图像膨胀操作确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # 距离运算确定前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # 寻找未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图像
    titles = ['原始图像', '阈值化', '开运算', '背景区域', '前景区域', '未知区域']
    images = [img, thresh, opening, sure_bg, sure_fg, unknown]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

def connectedComponentsWithStats():
    # 读入图片
    img = cv2.imread("res/bar_coin_3.png")
    # 中值滤波，去噪
    img = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('original', gray)

    # 阈值分割得到二值化图片
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 膨胀操作
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_clo = cv2.dilate(binary, kernel2, iterations=2)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_clo, connectivity=8)

    # 查看各个返回值
    # 连通域数量
    print('num_labels = ', num_labels)
    # 连通域的信息：对应各个轮廓的x、y、width、height和面积
    print('stats = ', stats)
    # 连通域的中心点
    print('centroids = ', centroids)
    # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
    print('labels = ', labels)

    # 不同的连通域赋予不同的颜色
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        mask = labels == i
        output[:, :, 0][mask] = np.random.randint(0, 255)
        output[:, :, 1][mask] = np.random.randint(0, 255)
        output[:, :, 2][mask] = np.random.randint(0, 255)
    cv2.imshow('oginal', output)
    cv2.waitKey()
    cv2.destroyAllWindows()

def testNDAssign():
    height, width = 3, 5
    output = np.zeros((height, width, 3), np.uint8)
    num_labels = np.zeros((height, width), np.uint8)
    for i in range(height):
        # num_labels[i,:] = i
        num_labels[i, 0:width] = i

    for i in range(height):
        mask = num_labels == i
        output[:, :, 0][mask] = (i + 1) * 20
        output[:, :, 1][mask] = (i + 1) * 20 + 1
        output[:, :, 2][mask] = (i + 1) * 20 + 2
    print(num_labels)
    print("")
    print(output)

# step3，当前处理结果中，已经能够区分出前景硬币区域和背景区域。接着
# 我们创建标记变量，在该变量中标记区域，已确认的区域（前景或背景）用不同
# 的正整数标记出来，不确认的区域保持 0 ， 使 用
# cv2.connectedComponents()函数来将图像背景标记成 0，其他目标用从
# 1 开始的整数标记。注意，如果背景被标记成 0，分水岭算法会认为它是未知区
# 域，所以要用不同的整数来标记。
# 最后，调用 watershed()函数实现分水岭图像分割，标记图像会被修改，
# 边界区域会被标记成 0
def watershed():
    # 读取原始图像
    img = cv2.imread('res/bar_coin_3.png')
    # 图像灰度化处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图像阈值化处理
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 图像开运算消除噪声
    kernal = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernal, iterations=2)
    # 图像膨胀操作确定背景区域
    sure_bg = cv2.dilate(opening, kernal, iterations=3)
    # 距离运算确定前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg =  cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # 寻找未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # 标记变量
    ret, markers = cv2.connectedComponents(sure_fg)
    # 所有标签加一，以确保背景不是 0 而是 1
    markers = markers + 1
    # 用 0 标记未知区域
    markers[unknown == 255] = 0
    # 分水岭算法实现图像分割
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图像
    titles = ['标记区域', '图像分割']
    images = [markers, img]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

# testNDAssign()
# connectedComponentsWithStats()
# threshold()
# morphologyEx()
watershed()