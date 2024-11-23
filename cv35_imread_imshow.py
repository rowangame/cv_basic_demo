
"""
OpenCV 读取图像的 imread()函数原型
如下，它将从指定的文件加载图像并返回矩阵，如果无法读取图像（因为缺少文
件 、 权 限 不 正 确 、 格 式 不 支 持 或 图 像 无 效 等 ） ， 则 返 回 空 矩 阵
（Mat::data-1==NULL）。

retval = imread(filename[, flags])
 filename 表示需要载入的图片路径名，其支持 Windows 位图、
JPEG 文件、PNG 图片、便携文件格式、Sun rasters 光栅文件、
TIFF 文件、HDR 文件等。
 flags 为 int 类型，表示载入标识，它指定一个加载图像的颜色类型，
默认值为 1。其中 cv2.IMREAD_UNCHANGED 表示读入完整图
像或图像不可变，包括 alpha 通道；cv2.IMREAD_GRAYSCALE
表示读入灰度图像；cv2.IMREAD_COLOR 表示读入彩色图像，默
认参数，忽略 alpha 通道。
"""

import cv2
import matplotlib.pyplot as plt

def test1():
    # 读取图片
    img = cv2.imread("res/lena_256.png")
    # 显示图像
    cv2.imshow("Demo", img)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test2():
    img = cv2.imread("res/lena_256.png")
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.imread("res/bar_barcode.png")
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.imread("res/sc_th.jpg")
    img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.imread("res/bar_window.png")
    img4 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    titles = ["lena", "barcode", "scene", "win-icon"]
    imgs = [img1, img2, img3, img4]
    for i in range(4):
        plt.subplot(2,2,i + 1)
        plt.imshow(imgs[i], "gray")
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    # plt.savefig("./out/cv35.png")
    plt.show()

test2()