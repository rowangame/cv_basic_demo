
import cv2
import numpy as np

"""
在OpenCV 中，绘制直线需要获取直线的起点和终点坐标，调用 cv2.line()
函数实现该功能。该函数原型如下所示：
 img = line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
 img 表示需要绘制的那幅图像
 pt1 表示线段第一个点的坐标
 pt2 表示线段第二个点的坐标
 color 表示线条颜色，需要传入一个 RGB 元组，如(255,0,0)代表蓝色
 thickness 表示线条粗细
 lineType 表示线条的类型
 shift 表示点坐标中的小数位数
"""
def drawLine():
    # 创建黑色图像
    img = np.zeros((256, 256, 3), np.uint8)
    # 绘制直线
    cv2.line(img, (0, 0), (255, 255), (55, 255, 155), 5)
    # 显示图像
    cv2.imshow("line", img)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
在 OpenCV 中，绘制矩形通过 cv2.rectangle()函数实现，该函数原型如
下所示：
 img = rectangle(img, pt1, pt2, color[, thickness[, lineType[,
shift]]])
 img 表示需要绘制的那幅图像
 pt1 表示矩形的左上角位置坐标
 pt2 表示矩形的右下角位置坐标
 color 表示矩形的颜色
 thickness 表示边框的粗细
 lineType 表示线条的类型
 shift 表示点坐标中的小数位数
"""
def drawRectangle():
    # 创建黑色图像
    img = np.zeros((256, 256, 3), np.uint8)
    # 绘制矩形
    cv2.rectangle(img, (20, 20), (150, 250), (255, 0, 0), 2)
    # 显示图像
    cv2.imshow("rectangle", img)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
 img = circle(img, center, radius, color[, thickness[,
lineType[, shift]]])
 img 表示需要绘制圆的图像
 center 表示圆心坐标
 radius 表示圆的半径
 color 表示圆的颜色
 thickness 如果为正值，表示圆轮廓的厚度；负厚度表示要绘制一
个填充圆
 lineType 表示圆的边界类型
 shift 表示中心坐标和半径值中的小数位数
"""
def drawCircle():
    # 创建黑色图像
    img = np.zeros((256, 256, 3), np.uint8)
    # 绘制圆形
    cv2.circle(img, (100, 100), 50, (255, 255, 0), 4)
    # 显示图像
    cv2.imshow("circle", img)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
在 OpenCV 中，绘制椭圆比较复杂，要多输入几个参数，如中心点的位置
坐标，长轴和短轴的长度，椭圆沿逆时针方向旋转的角度等。cv2.ellipse()函
数原型如下所示：
 img = ellipse(img, center, axes, angle, startAngle, endAngle,
color[, thickness[, lineType[, shift]]])
 img 表示需要绘制椭圆的图像
 center 表示椭圆圆心坐标
 axes 表示轴的长度（短半径和长半径）
 angle 表示偏转的角度（顺时针旋转）[与屏幕坐标系相关]
 startAngle 表示圆弧起始角的角度（顺时针旋转）[与屏幕坐标系相关]
 endAngle 表示圆弧终结角的角度（顺时针旋转）[与屏幕坐标系相关]
 color 表示线条的颜色
 thickness 如果为正值，表示椭圆轮廓的厚度；负值表示要绘制一
个填充椭圆
 lineType 表示圆的边界类型
 shift 表示中心坐标和轴值中的小数位数
"""
def drawEllipse():
    # 创建黑色图像
    img = np.zeros((256, 256, 3), np.uint8)
    # 绘制椭圆
    # 椭圆中心(120,100) 长轴和短轴为(100,20)
    # 偏转角度为 20
    # 圆弧起始角的角度 0 圆弧终结角的角度 360
    # 颜色(255,0,255) 线条粗细 2
    cv2.ellipse(img, (120, 100), (100, 30), 0, 0, 270, (255, 0, 0), 2)
    cv2.ellipse(img, (120, 100), (100, 50), 0, 45, 270, (255, 0,255), 2)
    # 显示图像
    cv2.imshow("ellipse", img)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
在 OpenCV 中，调用 cv2.polylines()函数绘制多边形，它需要指定每个
顶点的坐标，通过这些点构建多边形，其函数原型如下所示：
 img = polylines(img, pts, isClosed, color[, thickness[,
lineType[, shift]]])
 img 表示需要绘制的图像
 center 表示多边形曲线阵列
 isClosed 表示绘制的多边形是否闭合，False 表示不闭合
 color 表示线条的颜色
 thickness 表示线条粗细
 lineType 表示边界类型
 shift 表示顶点坐标中的小数位数
"""
def drawPolyLines():
    # 创建黑色图像
    img = np.zeros((256, 256, 3), np.uint8)
    # 绘制多边形
    pts = np.array([[10, 80], [120, 80], [120, 200], [180, 180]])
    cv2.polylines(img, [pts], True, (255, 255, 255), 2)
    # 显示图像
    cv2.imshow("poly-lines", img)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 画五角星
def drawFivePointStar():
    # 创建黑色图像
    img = np.zeros((512, 512, 3), np.uint8)
    # 绘制多边形
    pts = np.array([[50, 190], [380, 420], [255, 50], [120, 420],[450, 190]])
    cv2.polylines(img, [pts], True, (0, 255, 255), 2)
    # 显示图像
    cv2.imshow("five-point-star", img)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
在 OpenCV 中，调用 cv2.putText()函数添加对应的文字，其函数原型如
下所示：
 img = putText(img, text, org, fontFace, fontScale, color[,thickness[, lineType[, bottomLeftOrigin]]])
 img 表示要绘制的图像
 text 表示要绘制的文字
 org 表示要绘制的位置，图像中文本字符串的左下角
 fontFace 表示字体类型，具体查看 see cv::HersheyFonts
 fontScale 表示字体的大小，计算为比例因子乘以字体特定的基本大小
 color 表示字体的颜色
 thickness 表示字体的粗细
 lineType 表示边界类型
 bottomLeftOrigin 如果为真，则图像数据原点位于左下角，否则它在左上角
"""
def drawText():
    # 创建黑色图像
    img = np.zeros((256, 256, 3), np.uint8)
    # 绘制文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'I love Python!I love Cosonic!', (10, 100), font, 0.5, (255, 255, 0), 1)
    # 显示图像
    cv2.imshow("polylines", img)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# drawLine()
# drawRectangle()
drawCircle()
# drawEllipse()
# drawPolyLines()
# drawFivePointStar()
# drawText()