
import cv2

img = cv2.imread("res/lena_256.png")
cv2.rectangle(img, (10,10), (60, 60), (0, 255, 0), 2)
cv2.line(img, (80, 80), (220, 80), (255, 0, 255), 2)
cv2.imshow("rect", img)
cv2.waitKey(0)
cv2.destroyAllWindows()