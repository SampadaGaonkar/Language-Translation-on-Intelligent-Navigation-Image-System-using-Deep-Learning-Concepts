import numpy as np
import cv2
import pytesseract
import imutils
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# DEFINE FUNCTION
image_p1 = cv2.imread("penta pics/p1.jpg")
image_p2 = cv2.imread("penta pics/p2.jpg")
image_p3 = cv2.imread("penta pics/p3.jpg")
image_p4 = cv2.imread("penta pics/p4.jpg")
image_p5 = cv2.imread("penta pics/p5.jpg")
image_p6 = cv2.imread("penta pics/p6.jpg")
image_p7 = cv2.imread("penta pics/p7.jpg")
image_p8 = cv2.imread("penta pics/p8.jpg")
image_p9 = cv2.imread("penta pics/p9.jpg")
image_p10 = cv2.imread("penta pics/p10.jpg")
image_p11 = cv2.imread("penta pics/p11.jpg")

# Load the image
image_upload = cv2.imread("f.jpg")
# Resize the image
image = imutils.resize(image_upload, width=700)

# Original image display
cv2.imshow("Original image", image)
cv2.waitKey(0)


# Grayscale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Conversion", gray)
cv2.waitKey(0)

# Noise removal while preserving edges
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Bilateral filter", gray)
cv2.waitKey(0)

# find edges of grayscale conversion
edged = cv2.Canny(gray, 170, 300)
cv2.imshow("Canny Images", edged)
cv2.waitKey(0)

# Find contours
cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img1 = image.copy()
cv2.drawContours(img1, cnts, -1, (0, 255, 0), 3)
cv2.imshow("All contours", img1)
cv2.waitKey(0)

# MInimize the number of contours
cnts = sorted(cnts, key = cv2.contourArea, reverse= True)[:30]
NumberPlateCnt = None
img2 = image.copy()
cv2.drawContours(img2, cnts, -1, (0, 255, 0), 3)
cv2.imshow(" Selected Top contours", img2)
cv2.waitKey(0)


count = 0
idx =7
for c in cnts:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    if len(approx) == 5:
        Traffic_signCnt = approx

        x,y,w,h = cv2.boundingRect(c)
        new_img = image[y:y + h, x:x + w]
        cv2.imwrite("Cropped image containing text" + str(idx) + ".png", new_img)
        idx+= 1
    break



cv2.drawContours(image, [Traffic_signCnt], -1, (0,255,0), 3)
cv2.imshow("Final image with text to be extracted", image)
cv2.waitKey(0)

crop_img_loc = "Cropped image containing text7.png"
cv2.imshow("cropped image", cv2.imread(crop_img_loc))
cv2.waitKey(0)

# START OF EXTRACTION PROCESS

# Resize and grayscale the image
image1 = imutils.resize(cv2.imread(crop_img_loc), width=400)
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
cv2.imshow("1-Grayscale Conversion for cropped image", gray)
cv2.waitKey(0)

# Find Text-boxes in the image

h, w, c = image1.shape
boxes = pytesseract.image_to_boxes(image1)
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(image1, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

cv2.imshow('Textboxes in the image', image1)
cv2.waitKey(0)



if( image_upload.shape == image_p2.shape or image_upload.shape == image_p4.shape
        or image_upload.shape == image_p6.shape or image_upload.shape == image_p7.shape ):

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

elif(image_upload.shape == image_p1.shape or image_upload.shape == image_p3.shape or image_upload.shape == image_p5.shape or image_upload.shape == image_p8.shape
     or image_upload.shape == image_p9.shape  or image_upload.shape == image_p10.shape):

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
    text2 = re.findall(r'[A-Z][A-Za-z]+', text)
    print("The final text for translation :")
    for text2 in text2:
        print(text2, end=' ')

    cv2.waitKey(0)
else:
    print("different")





