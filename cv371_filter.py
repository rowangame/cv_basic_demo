
import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt

# 方框滤波
# 方框滤波又称为盒式滤波，它利用卷积运算对图像邻域的像素值进行平均处
# 理，从而实现消除图像中的噪声。方框滤波和和均值滤波的模糊内核基本一样，
# 区别为是否需要进行均一化处理
# dst = boxFilter(src, depth, ksize[, dst[, anchor[, normalize[,
# borderType]]]])
#  src 表示输入图像
#  dst 表示输出图像，其大小和类型与输入图像相同
#  depth 表示输出图像深度，通常设置为“-1”，表示与原图深度一致
#  ksize 表示模糊内核大小，以（宽度，高度）的形式呈现
#  normalize 表示是否对目标图像进行归一化处理，默认值为 true
#  anchor 表示锚点，即被平滑的那个点，其默认值 Point（-1，-1）表
# 示位于内核的中央，可省略
def boxFilter(normalize: bool):
    # 读取图片
    img = cv2.imread('res/lena_256.png')
    source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 方框滤波
    result = cv2.boxFilter(source, -1, (2, 2), normalize=normalize)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图形
    titles = ['原始图像', '方框滤波']
    images = [source, result]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

# dst = GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[,
# borderType]]])
#  src 表示待处理的输入图像
#  dst 表示输出图像，其大小和类型与输入图像相同
#  ksize 表示高斯滤波器模板大小，ksize.width 和 ksize.height 可以
# 不同，但它们都必须是正数和奇数，它们也可以是零，即（0, 0）
#  sigmaX 表示高斯核函数在 X 方向的高斯内核标准差
#  sigmaY 表示高斯核函数在 Y 方向的高斯内核标准差。如果 sigmaY
# 为零，则设置为等于 sigmaX，如果两个 sigma 均为零，则分别从
# ksize.width 和 ksize.height 计算得到
#  borderType 表示边框模式，用于推断图像外部像素的某种边界模式，
# 默认值为 BORDER_DEFAULT，可省略
def gaussianBlur():
    # 读取图片
    img = cv2.imread('res/lena_256.png')
    source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 高斯滤波
    result = cv2.GaussianBlur(source, (7, 7), 0)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图形
    titles = ['原始图像', '高斯滤波']
    images = [source, result]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

# 中值滤波通过计算
# 每一个像素点某邻域范围内所有像素点灰度值的中值，来替换该像素点的灰度值，
# 从而让周围的像素值更接近真实情况，消除孤立的噪声
def mediaBlur():
    img = cv2.imread('res/lena_256_ns2.png')
    source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 中值滤波
    result = cv2.medianBlur(source, 3)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图形
    titles = ['原始图像', '中值滤波']
    images = [source, result]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

# 生成一张噪声图
def makeNoisePic():
    imgTmp = cv2.imread('res/lena_256.png')
    img = cv2.cvtColor(imgTmp, cv2.COLOR_BGR2RGB)
    ncount = 350

    height, width = img.shape[:2]
    # 复制图像数据
    # newImg = img.copy()
    # 或者创建图像数据并赋值
    newImg = np.zeros(img.shape, np.uint8)
    newImg[0:height, 0:width] = img[0:height, 0:width]
    # 随机生成白色的噪声点
    rndX = np.random.randint(width, size=ncount)
    rndY = np.random.randint(height, size=ncount)
    # 噪声半径范围(1,3)
    radius = np.random.randint(3, size=ncount) + 1
    for i in range(ncount):
        tmpR = radius[i]
        if tmpR == 1:
            newImg[rndY[i],rndX[i]] = 255
        else:
            # newImg[rndY[i], rndX[i]] = 255
            # 生成一个区域半径数组(值由-radius/2 ... radius/2)
            tmpH = tmpR * 2 - 1
            tmpW = tmpR * 2 - 1
            tmpDelta = (tmpR * 2 - 1) // 2
            tmpArea = np.zeros((tmpH, tmpW, 2), np.uint8)
            for ty in range(tmpH):
                for tx in range(tmpW):
                    tmpArea[ty, tx] = [ty - tmpDelta, tx - tmpDelta]
            # 赋值噪声数据
            for ty in range(tmpH):
                for tx in range(tmpW):
                    refY = rndY[i] + tmpArea[ty,tx][0]
                    refX = rndX[i] + tmpArea[ty,tx][1]
                    if ((refY >= 0) and (refY < height)) and ((refX >= 0) and (refX < width)):
                        newImg[refY, refX] = 255
    # 保存图像(因cv2保存文件中,数据是BGR格式,所以需要转换一下)
    desImg = cv2.cvtColor(newImg, cv2.COLOR_RGB2BGR)
    cv2.imwrite("res/lena_256_ns1.png", desImg)

    # 显示图形(plt.imshow显示图像是RGB格式,所以初始时需要转换一下)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    titles = ['原始图像', '噪声图像']
    images = [img, newImg]
    for i in range(2):
        plt.subplot(1,2, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

# 双边滤波（Bilateral filter）是由 Tomasi 和 Manduchi 在 1998 年发
# 明的一种各向异性滤波，它一种非线性的图像平滑方法，结合了图像的空间邻近
# 度和像素值相似度（即空间域和值域）的一种折中处理，从而达到保边去噪的目
# 的。双边滤波的优势是能够做到边缘的保护，其他的均值滤波、方框滤波和高斯
# 滤波在去除噪声的同时，都会有较明显的边缘模糊，对于图像高频细节的保护效
# 果并不好（文档参考：388页）
def bilateralFilter():
    # 读取图片
    img = cv2.imread('res/lena_256_ns2.png')
    source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 双边滤波
    result = cv2.bilateralFilter(source, 15, 50, 50)
    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 显示图形
    titles = ['原始图像', '双边滤波']
    images = [source, result]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

# makeNoisePic()
# boxFilter(False)
# boxFilter(True)
# gaussianBlur()
# mediaBlur()
bilateralFilter()