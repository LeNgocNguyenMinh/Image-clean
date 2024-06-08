# (1) Convert to binary image (mask of numbers) using thresholding techniques
# (2) Using opening and closing to remove noise and connect broken digits
# (3) Find contours and bounding boxes (using methods in Contour features)

import cv2 as cv
import numpy as np

# Read image
img = cv.imread('digits.png')
gray_img = cv.imread('digits.png', 0)

# Get threshold
th1 = cv.adaptiveThreshold(gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
_,th2 = cv.threshold(th1, 0, 255, cv.THRESH_BINARY_INV)

# Mask 1 (5)
mask_1 = th2[250:294, 254:274]

# Handle mask 1
mask_1_kernel = np.ones((2,2), np.uint8)
mask_1_handled = cv.erode(mask_1, mask_1_kernel, iterations = 1)

# Mask 2 (4) [124, 186], [145, 238]
mask_2 = th2[232:240, 128:140]

# Handle mask 2
mask_2_kernel = np.ones((4,2), np.uint8)
mask_2_handled = cv.erode(mask_2, mask_2_kernel, iterations = 1)

# Mask 3 (4) [10, 246], [10, 298]
mask_3 = th2[246:306, :10]

# Handle mask 2
mask_3_kernel_1 = np.ones((0,3), np.uint8)
mask_3_handled = cv.dilate(mask_3, mask_3_kernel_1, iterations = 1)
mask_3_kernel_2 = np.ones((3,2), np.uint8)
mask_3_handled = cv.erode(mask_3_handled, mask_3_kernel_2, iterations = 3)

# Noise area
noise_area = th2[160:, 168:]

# Handle noise area
kernel_1 = np.ones((2,3), np.uint8)
noise_area_handled = cv.morphologyEx(noise_area, cv.MORPH_OPEN, kernel_1, iterations=1)
kernel_2 = np.ones((3,2), np.uint8)
noise_area_handled = cv.erode(noise_area_handled, kernel_2, iterations = 1)

# Handle image
kernel_3 = np.ones((2,1), np.uint8)
opening_2 = cv.morphologyEx(th2, cv.MORPH_OPEN, kernel_3, iterations=1)
opening_2 = cv.erode(opening_2, np.ones((1,1), np.uint8) ,iterations = 1)

opening_2[160:, 168:] = noise_area_handled
opening_2[250:294, 254:274] = mask_1_handled
opening_2[232:240, 128:140] = mask_2_handled
opening_2[246:306, :10] = mask_3_handled

kernel_4 = np.ones((2,1), np.uint8)
closing_2 = cv.morphologyEx(opening_2, cv.MORPH_CLOSE, kernel_4)

# Find contours
contours, hierarchy  = cv.findContours(image=closing_2, mode = cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

# Draw rectanges
for contour in contours:
    x, y, w, h = cv.boundingRect(contour)
    if h > 33 and (w > 0 and w < 60):
        img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show image
cv.imshow("Image", img)
cv.waitKey(0)
cv.destroyAllWindows()

# Write image
cv.imwrite("mydigits.jpg", img)