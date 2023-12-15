from PIL import Image
import pytesseract
import cv2
import numpy as np

# Load the image from file
image = cv2.imread('suc.jpeg')

# Rescale the image
image = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply dilation and erosion to remove some noise
kernel = np.ones((1, 1), np.uint8)
gray = cv2.dilate(gray, kernel, iterations=1)
gray = cv2.erode(gray, kernel, iterations=1)

# Apply threshold to get image with only black and white
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Save the preprocessed image temporarily
cv2.imwrite('temp.png', thresh)

# Open the image with PIL so pytesseract can use it
image = Image.open('temp.png')

# Use pytesseract to do OCR on the image
text = pytesseract.image_to_string(image,lang='tur')

# Print the text
print(text)