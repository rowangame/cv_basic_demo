
import cv2
import numpy as np

def copyImg():
    # 读取图片
    img = cv2.imread("res/lena_256.png")
    # 创建空图像
    emptyImage = np.zeros(img.shape, np.uint8)
    # 复制图像
    emptyImage2 = img.copy()
    # 显示图像
    cv2.imshow("Demo1", img)
    cv2.imshow("Demo2", emptyImage)
    cv2.imshow("Demo3", emptyImage2)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
在 OpenCV 中，输出图像到文件使用的函数为 imwrite()，其函数原型如
下：
retval = imwrite(filename, img[, params])
 filename 表示要保存的路径及文件名
 img 表示图像矩阵
 params 表示特定格式保存的参数编码，默认值为空。对于 JPEG 图
片，该参数（cv2.IMWRITE_JPEG_QUALITY）表示图像的质量，
用 0-100 的整数表示，默认值为 95。对于 PNG 图片，该参数
（cv2.IMWRITE_PNG_COMPRESSION）表示的是压缩级别，
从 0 到 9，压缩级别越高，图像尺寸越小，默认级别为 3。对于 PPM、
PGM 、 PBM 图 片 ， 该 参 数 表 示 一 个 二 进 制 格 式 的 标 志
（cv2.IMWRITE_PXM_BINARY）[2]。注意，该类型为 Long，必
须转换成 int。
"""
def saveImg():
    # 读取图像
    img = cv2.imread("res/lena_256.png")
    # 显示图像
    cv2.imshow("Demo", img)

    # 保存图像
    cv2.imwrite("./out/dst1.jpg", img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 5])
    cv2.imwrite("./out/dst2.jpg", img,
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite("./out/dst3.png", img,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.imwrite("./out/dst4.png", img,
                [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# copyImg()
saveImg()