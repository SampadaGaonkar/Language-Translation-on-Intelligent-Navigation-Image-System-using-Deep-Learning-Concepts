import cv2
import numpy as np
import pytesseract
import imutils
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# DEFINE FUNCTION
image_c1 = cv2.imread("shapes/c1.jpg")
image_c2 = cv2.imread("shapes/c2.jpg")
image_c6 = cv2.imread("shapes/c6.jpg")


img1 = cv2.imread('f.jpg')
img = cv2.imread('f.jpg',0)
gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# Create mask
height,width = img.shape
mask = np.zeros((height,width), np.uint8)

edges = cv2.Canny(thresh, 100, 200)
#cv2.imshow('detected ',gray)
cimg=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
for i in circles[0,:]:
    i[2]=i[2]+4
    # Draw on mask
    cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)

# Copy that image using that mask
masked_data = cv2.bitwise_and(img1, img1, mask=mask)

# Apply Threshold
_,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

# Find Contour
contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
x,y,w,h = cv2.boundingRect(contours[0])

# Crop masked_data
crop = masked_data[y:y+h,x:x+w]

#Code to close Window
cv2.imshow('detected Edge',img1)
cv2.waitKey()
cv2.imshow('Cropped Eye',crop)
cv2.waitKey(0)


######################################################################################

if( img1.shape == image_c2.shape ):
    # Resize and grayscale the image
    image1 = imutils.resize(crop, width=200)



    # Extract the text
    text = pytesseract.image_to_string(image1, lang="eng")
    print(" The Extracted text  is: \n", text)
    print("\n")

    # Remove the numbers
    custom_config = r'-c tessedit_char_blacklist=0123456789 --psm 6'
    text1 = pytesseract.image_to_string(image1, config=custom_config)
    print("The Extracted text excluding numbers: \n", text1)
    print("\n")

    # Remove all special characters and display final text
    text2 = re.findall(r'[A-Z][A-Za-z]+', text)
    print("The final text for translation :")
    for text2 in text2:
       print(text2, end=' ')

    cv2.waitKey(0)

elif(img1.shape == image_c1.shape):
    # Resize and grayscale the image
    image1 = imutils.resize(crop, width=490)

    # Grayscale
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    cv2.imshow("1-Grayscale Conversion for cropped image", gray)
    cv2.waitKey(0)

    # Removal of noise
    noise = cv2.blur(gray, (3, 3))
    cv2.imshow("Noise reduction for cropped image", gray)
    cv2.waitKey(0)

    # thresholding the image
    thres = cv2.threshold(noise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imshow("Thresholding  for cropped image", thres)
    cv2.waitKey(0)

    # opening
    kernel = np.ones((2, 2), np.uint8)
    open = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)
    cv2.imshow("opening cropped image", open)
    cv2.waitKey(0)

    # Extract the text
    text = pytesseract.image_to_string(open, lang="eng")
    print(" The Extracted text  is: \n", text)
    print("\n")

    # Remove the numbers
    custom_config = r'-c tessedit_char_blacklist=0123456789 --psm 6'
    text1 = pytesseract.image_to_string(open, config=custom_config)
    print("The Extracted text excluding numbers: \n", text1)
    print("\n")

    # Remove all special characters and display final text
    text2 = re.findall(r'[A-Z]+', text)
    print("The final text for translation :")
    for text2 in text2:
        print(text2, end=' ')

    cv2.waitKey(0)
else:
    # Resize and grayscale the image
    image1 = imutils.resize(crop, width=200)

    # Grayscale
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    cv2.imshow("1-Grayscale Conversion for cropped image", gray)
    cv2.waitKey(0)

    # Extract the text
    text = pytesseract.image_to_string(gray, lang="eng")
    print(" The Extracted text  is: \n", text)
    print("\n")

    # Remove the numbers
    custom_config = r'-c tessedit_char_blacklist=0123456789 --psm 6'
    text1 = pytesseract.image_to_string(gray, config=custom_config)
    print("The Extracted text excluding numbers: \n", text1)
    print("\n")

    # Remove all special characters and display final text
    text2 = re.findall(r'[A-Z][A-Za-z]+', text)
    print("The final text for translation :")
    for text2 in text2:
        print(text2, end=' ')

    cv2.waitKey(0)