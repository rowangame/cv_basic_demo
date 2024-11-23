"""
自动色彩均衡化
Retinex 算法是代表性的图像增强算法，它根据人的视网膜和大脑皮层模拟
对物体颜色的波长光线反射能力而形成，对复杂环境下的一维条码具有一定范围
内的动态压缩，对图像边缘有着一定自适应的增强。自动色彩均衡（Automatic
Color Enhancement，ACE）算法是在 Retinex 算法的理论上提出的，它
通过计算图像目标像素点和周围像素点的明暗程度及其关系来对最终的像素值
进行校正，实现图像的对比度调整，产生类似人体视网膜的色彩恒常性和亮度恒
常性的均衡，具有很好的图像增强效果

ACE 算法包括两个步骤，
一是对图像进行色彩和空域调整，完成图像的色差校正，得到空域重构图像；
二是对校正后的图像进行动态扩展。
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# 线性拉伸处理
# 去掉最大最小 0.5%的像素值 线性拉伸至[0,1]
def stretchImage(data, s=0.005, bins = 2000):
    ht = np.histogram(data, bins)
    d = np.cumsum(ht[0]) / float(data.size)
    lmin = 0
    lmax = bins - 1
    while lmin < bins:
        if d[lmin] >= s:
            break
        lmin += 1
    while lmax >= 0:
        if d[lmax] <= 1 - s:
            break
        lmax -= 1
    return np.clip((data-ht[1][lmin]) / (ht[1][lmax]- ht[1][lmin]), 0, 1)

#根据半径计算权重参数矩阵
g_para = {}
def getPara(radius = 5):
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius * 2 + 1
    m = np.zeros((size, size))
    for h in range(-radius, radius + 1):
        for w in range(-radius, radius + 1):
            if h == 0 and w == 0:
                continue
            m[radius + h, radius + w] = 1.0 / math.sqrt(h ** 2 + w ** 2)
    m /= m.sum()
    g_para[radius] = m
    return m

#常规的 ACE 实现
def zmIce(I, ratio=4, radius=300):
    para = getPara(radius)
    height, width = I.shape

    # Python3 报错如下 使用列表 append 修改
    zh = []
    zw = []
    n = 0
    while n < radius:
        zh.append(0)
        zw.append(0)
        n += 1
    for n in range(height):
        zh.append(n)
    for n in range(width):
        zw.append(n)
    n = 0
    while n < radius:
        zh.append(height - 1)
        zw.append(width - 1)
        n += 1
    # print(zh)
    # print(zw)
    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip((I - Z[h:h + height, w:w + width]) * ratio, -1, 1))
    return res

#单通道 ACE 快速增强实现
def zmIceFast(I, ratio, radius):
    # print(I)
    height, width = I.shape[:2]
    if min(height, width) <= 2:
        return np.zeros(I.shape) + 0.5
    Rs = cv2.resize(I, (int((width + 1) / 2), int((height + 1) / 2)))
    Rf = zmIceFast(Rs, ratio, radius)  # 递归调用
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))
    return Rf + zmIce(I, ratio, radius) - zmIce(Rs, ratio, radius)

# rgb:三通道分别增强 ratio:对比度增强因子 radius:卷积模板半径
def zmIceColor(I, ratio=4, radius=3):
    res = np.zeros(I.shape)
    for k in range(3):
        res[:, :, k] = stretchImage(zmIceFast(I[:, :, k], ratio, radius))
    return res

# ace测试算法通过
def testAce():
    # img = cv2.imread('./res/sc_foggy_1.png')
    img = cv2.imread('res/sc_foggy_2.png')
    res = zmIceColor(img / 255.0) * 255
    # 保存文件
    # cv2.imwrite("./out/foggy-ace.jpg", res)

    # 显示时需要将类型转换(否则显示内容会出问题)
    resInt = res.astype(np.uint8)
    cv2.imshow("Input", img)
    cv2.imshow("Result", resInt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 矩阵数据赋值是从下标值开始(包含此值),下标结束(不包含此值)
def testNdarray():
    """
    data-1 = np.zeros((3,3), np.uint8)
    print(data-1.shape, data-1)
    h,w = data-1.shape[:2]
    print(h, w)

    print("assign value...")
    data-1[0:1, 0:2] = 255
    print(data-1.shape)
    print(data-1)

    d2 = np.arange(0, 10)
    print(d2.shape, d2)
    d2[1:4] = 5
    # 这里区分stop值为负的情况[表示从数据尾部数(尾部数从1开始),第几个结束]
    # d2[1:-2] = 5
    print(d2)
    """

    # """
    # # 字符串的下标运算也是一样(>=start,<stop)
    # str = "Hello"
    # print("len(str)=", len(str), str)
    # print(str[0:3])
    # """

    print("ndarray minus:")
    da = np.arange(0, 12)
    da = da.reshape(3, 4)
    db = np.ones((3, 4), np.uint8)
    print(da.shape, da)
    print(db.shape, db)
    # 这里减少运算，需要da.shape == db.shape才行
    dc = da - db
    print(dc)

    print("ndarray times:")
    de = da * db
    print(de)

def testHistogram():
    rnd = np.random.rand(10)
    print(type(rnd), rnd)
    # a是待统计数据的数组；
    # bins指定统计的区间个数；
    # range是一个长度为2的元组，表示统计范围的最小值和最大值，默认值None，表示范围由数据的范围决定
    # weights为数组的每个元素指定了权值, histogram()
    # 会对区间中数组所对应的权值进行求和
    # density为True时，返回每个区间的概率密度；为False，返回每个区间中元素的个数
    hist, bins = np.histogram(rnd, bins=5, range=(0, 1))
    print(hist.shape, hist, bins.shape, bins)
    hist = np.histogram(rnd, bins=5, range=(0, 1))
    print(hist)

# cumsum的作用主要就是计算轴向的累加和
def testCumsum():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    print(a, "sum(a)=", a.sum())

    rlt = np.cumsum(a)
    print(type(rlt), rlt.shape, rlt)

    print("按行累加 np.cumsum axis=0")
    rlt = np.cumsum(a, axis=0) # 按行累加
    print(type(rlt), rlt.shape, rlt)

    print("按列累加 np.cumsum axis=1")
    rlt = np.cumsum(a, axis=1) # 按列累加
    print(type(rlt), rlt.shape, rlt)

def testClip():
    a = np.arange(10)
    rlt = np.clip(a, 1, 5)
    print(a.shape, a)
    print(rlt.shape, rlt)

def testAssignByMultiLines():
    a = np.arange(15)
    aX = a.reshape(3, 5)
    print(a.shape, aX.shape)
    print(aX)

    tmpIdx = np.ix_([1, 2], [1, 3, 4])
    print(type(tmpIdx), len(tmpIdx))
    print(tmpIdx)
    # for tmpV in tmpIdx:
    #     print("shape=", tmpV.shape, "value=", tmpV)

    print("after assign:")
    aX[tmpIdx] = -1
    print(aX)
    print("after get:")
    rlt = aX[tmpIdx]
    print(rlt)

#主函数
if __name__ == '__main__':
    # testNdarray()
    # testHistogram()
    # testCumsum()
    # testClip()
    # testAssignByMultiLines()
    testAce()