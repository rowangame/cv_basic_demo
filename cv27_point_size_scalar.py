from cv2.typing import Point,Size,Scalar

p: Point = (1, 2) * 3
print(p, type(p))

points_list = [(160, 160), (136, 160)]
print(points_list, type(points_list), type(points_list[0]))

s : Size = (3, 5)
print(s)

sl : Scalar = (3,4,5)
print(sl)