"""
一种用来判断图像是白天还是黑夜的方法，其基本步骤如下：
（1）读取原始图像，转换为灰度图，并获取图像的所有像素值；
（2）设置灰度阈值并计算该阈值以下的像素个数。比如像素的阈值设置为
50，统计低于 50 的像素值个数；
（3）设置比例参数，对比该参数与低于该阈值的像素占比，如果低于参数
则预测为白天，高于参数则预测为黑夜。比如该参数设置为 0.8，像素的灰度值
低于阈值 50 的个数占整幅图像所有像素个数的 90%，则认为该图像偏暗，故
322预测为黑夜；否则预测为白天
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

#函数: 判断黑夜或白天
def func_judge(img):
    # 获取图像高度和宽度
    height = img.shape[0]
    width = img.shape[1]
    piexs_sum = height * width
    dark_sum = 0  # 偏暗像素个数
    dark_prop = 0  # 偏暗像素所占比例
    for i in range(height):
        for j in range(width):
            if img[i, j] < 50:  # 阈值为 50
                dark_sum += 1
    # 计算比例
    print(dark_sum)
    print(piexs_sum)
    dark_prop = dark_sum * 1.0 / piexs_sum
    if dark_prop >= 0.8:
        print("This picture is dark!", dark_prop)
    else:
        print("This picture is bright!", dark_prop)


def testDayNight(srcpath):
    img = cv2.imread(srcpath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 计算 256 灰度级的图像直方图
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 255])
    # 判断黑夜或白天
    func_judge(gray_img)
    # 显示原始图像和绘制的直方图
    plt.subplot(121)
    plt.imshow(img_rgb, 'gray')
    plt.axis("off")
    plt.title("(a)")

    plt.subplot(122)
    plt.plot(hist, color='r')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("(b)")

    plt.show()

# 简单的通过阈值判断白天和黑夜,并不能应用于实际的复杂场景中(仅仅作为测试用）
# 人类视觉判断白天和黑夜是相当复杂的过程，根据时间，太阳月亮下山，还有当前处在环境中，
# 如果夜晚有各种环境光源，人类也能很快分析白天和黑夜等场景。
# 实际应用需要分析场景内容，地理位置，光源分析，时间分析才能给出一个准确的判断来
# srcpath = "./res/sc_day.jpg"
srcpath = "./res/sc_night.jpg"
testDayNight(srcpath)




