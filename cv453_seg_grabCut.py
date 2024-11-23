"""
GrabCut 算法原型如下所示：
mask, bgdModel, fgdModel = grabCut(img, mask, rect,
bgdModel, fgdModel, iterCount[, mode])
 image 表示输入图像，为 8 位三通道图像
 mask 表示蒙板图像，输入/输出的 8 位单通道掩码，确定前景区
域、背景区域、不确定区域。当模式设置为GC_INIT_WITH_RECT 时，该掩码由函数初始化
 rect 表示前景对象的矩形坐标，其基本格式为(x, y, w, h)，分别为左上角坐标和宽度、高度
 bdgModel 表示后台模型使用的数组，通常设置为大小为（1, 65）
np.float64 的数组
 fgdModel 表示前台模型使用的数组，通常设置为大小为（1, 65）
np.float64 的数组
 iterCount 表示算法运行的迭代次数
 mode 是 cv::GrabCutModes 操作模式之一，
cv2.GC_INIT_WITH_RECT 或
cv2.GC_INIT_WITH_MASK 表示使用矩阵模式或蒙板模式
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#读取图像
img = cv2.imread('res/sc_night.jpg')
#灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#设置掩码、fgbModel、bgModel
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
#矩形坐标
rect = (460, 235, 600, 345)
#图像分割
cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
#设置新掩码：0 和 2 做背景
mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

#设置字体
matplotlib.rcParams['font.sans-serif']=['SimHei']

#显示原图
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.imshow(img)
plt.title('(a)原始图像')
plt.xticks([]), plt.yticks([])

#使用蒙板来获取前景区域
img = img*mask2[:, :, np.newaxis]
plt.subplot(1,2,2)
plt.imshow(img)
plt.title('(b)目标图像')

plt.colorbar()
plt.xticks([]), plt.yticks([])
plt.show()




