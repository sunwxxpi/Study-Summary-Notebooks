from PIL import Image

image_path = 'sunwxxpi.png'
image = Image.open(image_path)

image.save('saved_image.png')
image.show()

# 'L'은 8-bit Grayscale 이미지를 의미한다.
# '.convert('RGB')'를 사용하면 RGB 이미지로 변환할 수 있다.
gray_image = image.convert('L')

gray_image.save('gray_image.png')
gray_image.show() # File Viewer에 따라 RGB 이미지로 표시될 수도 있다. (실제로는 Grayscale)