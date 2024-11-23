
import cv2
import numpy as np

def process_1():
    # 读取图片
    img = cv2.imread("res/lena_256.png")
    # 读取像素
    test = img[88, 142]
    print(type(test), id(test), test.shape)
    print("读取的像素值:", test)
    # 修改像素
    img[88, 142] = [255, 255, 255]
    print("修改后的像素值:", test)
    # 分别获取 BGR 通道像素
    blue = img[88, 142, 0]
    print("蓝色分量", blue)
    green = img[88, 142, 1]
    print("绿色分量", green)
    red = img[88, 142, 2]
    print("红色分量", red)
    # 显示图像
    cv2.imshow("Demo", img)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 将 100 到 200 行、150 到 250 列的像素区域设置为白色的效果。
def process_2():
    # 读取图片
    img = cv2.imread("res/lena_256.png")
    img[100:200, 150:250] = [255,255,255]
    # 显示图像
    cv2.imshow("Demo", img)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
在图像处理中，NumPy 读取像素调用 item()函数实现，修改像素调用
itemset()实现，其原型如下所示[5]。使用 Numpy 进行像素读取，调用方式如下
"""
def process_itemset():
    # 读取图片
    img = cv2.imread("res/lena_256.png")
    print(type(img))
    # Numpy 读取像素
    print(img.item(78, 100, 0))
    print(img.item(78, 100, 1))
    print(img.item(78, 100, 2))
    # Numpy 修改像素
    img.itemset((78, 100, 0), 100)
    img.itemset((78, 100, 1), 100)
    img.itemset((78, 100, 2), 100)

"""
由于在 OpenCV2 中没有 CreateImage 函数，如果需要创建图像，则需
要使用 Numpy 库函数实现。如下述代码，调用 np.zeros()函数创建空图像，
创建的新图像使用 Numpy 数组的属性来表示图像的尺寸和通道信息，其中参
数 img.shape 表示原始图像的形状，np.uint8 表示类型
"""
def process_create_image():
    # 读取图片
    img = cv2.imread("res/lena_256.png")
    # 创建空图像
    emptyImage = np.zeros(img.shape, np.uint8)
    # 显示图像
    cv2.imshow("Demo", emptyImage)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# process_1()
process_2()
# process_itemset()
# process_create_image()