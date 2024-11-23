"""
图像漫水填充分割实现
主要通过 floodFill()函数实现漫水填充分割，它将用指定
的颜色从种子点开始填充一个连接域。其函数原型如下所示：
floodFill(image, mask, seedPoint, newVal[, loDiff[, upDiff[,
flags]]])
 image 表示输入/输出 1 通道或 3 通道，6 位或浮点图像
 mask 表示操作掩码，必须为 8 位单通道图像，其长宽都比输入图
像大两个像素点。注意，漫水填充不会填充掩膜 mask 的非零像素
区域，mask 中与输入图像(x,y)像素点相对应的点的坐标为
(x+1,y+1)。
 seedPoint 为 Point 类型，表示漫水填充算法的起始点
 newVal 表示像素点被染色的值，即在重绘区域像素的新值

 loDiff 表示当前观察像素值与其部件邻域像素值或待加入该部件的
种子像素之间的亮度或颜色之负差的最大值，默认值为 Scalar( )
 upDiff 表示当前观察像素值与其部件邻域像素值或待加入该部件
的种子像素之间的亮度或颜色之正差的最大值，默认值为 Scalar( )
 flags 表示操作标识符，此参数包括三个部分：低八位 0-7bit 表示
邻接性（4 邻接或 8 邻接）；中间八位 8-15bit 表示掩码的填充颜
色，如果中间八位为 0 则掩码用 1 来填充；高八位 16-31bit 表示
填 充 模 式 ， 可 以 为 0 或 者 以 下 两 种 标 志 符 的 组 合 ，
FLOODFILL_FIXED_RANGE 表示此标志会考虑当前像素与
种子像素之间的差，否则就考虑当前像素与 相邻像素的差。
FLOODFILL_MASK_ONLY 表示函数不会去填充改变原始图
像,而是去填充掩码图像 mask，mask 指定的位置为零时才填充，
不为零不填充。
"""

import cv2
import numpy as np

def floodFill():
    # 读取原始图像
    img = cv2.imread('res/bar_window.png')
    # 获取图像行和列
    rows, cols = img.shape[:2]
    # 目标图像
    dst = img.copy()
    # mask 必须行和列都加 2 且必须为 uint8 单通道阵列
    # mask 多出来的 2 可以保证扫描的边界上的像素都会被处理
    mask = np.zeros([rows + 2, cols + 2], np.uint8)
    # 图像漫水填充处理
    # 种子点位置(30,30) 设置颜色(0,255,255) 连通区范围设定 loDiff, upDiff
    # src(seed.x, seed.y) - loDiff <= src(x, y) <= src(seed.x, seed.y) +upDiff
    cv2.floodFill(dst, mask, (30, 30), (0, 255, 255),
                  (100,100,100), (50, 50, 50),
                  cv2.FLOODFILL_FIXED_RANGE)
    # 显示图像
    cv2.imshow("src", img)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

floodFill()