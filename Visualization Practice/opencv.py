import cv2

# OpenCV는 BGR 형식으로 이미지를 처리한다.
# 이미지 저장, 이미지 시각화 등, 모든 작업을 OpenCV를 통해 수행한다면, RGB 형식으로 변환할 필요가 없다.

image_path = 'sunwxxpi.png'
image = cv2.imread(image_path)

cv2.imwrite('saved_image.png', image)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('gray_image.png', gray_image)
cv2.imshow('Gray Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()