"""
在 OpenCV 中，可以通过
cv2.findContours()函数从二值图像中寻找轮廓，其函数原型如下所示
image, contours, hierarchy = findContours(image, mode,
method[, contours[, hierarchy[, offset]]])
 image 表示输入图像，即用于寻找轮廓的图像，为 8 位单通道
 contours 表示检测到的轮廓，其函数运行后的结果存在该变量中
每个轮廓存储为一个点向量
 hierarchy 表示输出变量，包含图像的拓扑信息，作为轮廓数量的
表示，它包含了许多元素，每个轮廓 contours[i]对应 4 个
hierarchy 元素 hierarchy[i][0]至 hierarchy[i][3]，分别表示后
一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号
 mode 表示轮廓检索模式。cv2.RETR_EXTERNAL 表示只检
测外轮廓；cv2.RETR_LIST 表示提取所有轮廓，且检测的轮廓不
建立等级关系；cv2.RETR_CCOMP 提取所有轮廓，并建立两个
等级的轮廓，上面的一层为外边界，里面一层为内孔的边界信；
cv2.RETR_TREE 表示提取所有轮廓，并且建立一个等级树或网
状结构的轮廓
 method 表示轮廓的近似方法。cv2.CHAIN_APPROX_NONE
存储所有的轮廓点，相邻的两个点的像素位置差不超过 1，即 max
（ abs(x1-x2), abs(y1-y2) ） =1 ；
cv2.CHAIN_APPROX_SIMPLE 压缩水平方向、垂直方向、对
角线方向的元素，只保留该方向的终点坐标，例如一个矩阵轮廓只
需 4 个点来保存轮廓信息；cv2.CHAIN_APPROX_TC89_L1
和 cv2.CHAIN_APPROX_TC89_KCOS 使用 Teh-Chinl
Chain 近似算法
 offset 表示每个轮廓点的可选偏移量

在使用 findContours() 函 数 检 测 图 像 边 缘 轮 廓 后 ， 通 常 需 要 和
drawContours()函数联合使用，接着绘制检测到的轮廓，drawContours()
函数的原型如下：
image = drawContours(image, contours, contourIdx,
color[, thickness[, lineType[, hierarchy[, maxLevel[,
offset]]]]])
 image 表示目标图像，即所要绘制轮廓的背景图片
 contours 表示所有的输入轮廓，每个轮廓存储为一个点向量
 contourldx 表示轮廓绘制的指示变量，如果为负数表示绘制所有轮廓
 color 表示绘制轮廓的颜色
 thickness 表里绘制轮廓线条的粗细程度，默认值为 1
 lineType 表示线条类型，默认值为 8，可选线包括 8（8 连通线型）、4（4 连通线型）、CV_AA（抗锯齿线型）
 hierarchy 表示可选的层次结构信息
 maxLevel 表示用于绘制轮廓的最大等级，默认值为 INT_MAX
 offset 表示每个轮廓点的可选偏移量
"""

import cv2


#读取图像
img = cv2.imread('res/sc_th.jpg')
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#阈值化处理
ret, binary = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#边缘检测
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#轮廓绘制(第四个参数:BGR颜色)
cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

#显示图像
cv2.imshow('gray', binary)
cv2.imshow('res', img)
cv2.waitKey(0)
cv2.destroyAllWindows()