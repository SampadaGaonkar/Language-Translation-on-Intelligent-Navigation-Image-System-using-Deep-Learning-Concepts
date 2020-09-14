from django.shortcuts import render
from django.conf import settings
import numpy as np
import cv2
import pytesseract
import imutils
import re
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 

def opencv(path):
    image_upload = cv2.imread(path, 1)
    
    image_s1 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/s1.jpg")
    image_s2 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/s2.jpg")
    image_s3 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/s3.jpg")
    image_s4 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/s4.jpg")
    image_s5 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/s5.jpg")
    image_f1 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/f1.jpg")
    image_f2 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/f2.jpg")
    image_f3 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/f3.jpg")
    image_f4 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/f4.jpg")
    image_f5 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/f5.jpg")
    image_f6 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/f6.jpg")
    image_f7 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/f7.jpg")
    image_f8 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/f8.jpg")
    image_f9 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/f9.jpg")
    image_f10 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/f10.jpg")
    image_f11 = cv2.imread("D:/BE Main project/djangoprojects/Projects/Beproject/test_images/f11.jpg")

    if (type(image_upload) is np.ndarray):
        print(image_upload.shape)

        image = imutils.resize(image_upload, width=700)

        # Grayscale conversion
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Noise removal while preserving edges
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # find edges of grayscale conversion
        edged = cv2.Canny(gray, 170, 300)

        # Find contours
        cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        img1 = image.copy()
        cv2.drawContours(img1, cnts, -1, (0, 255, 0), 3)

        # MInimize the number of contours
        cnts = sorted(cnts, key = cv2.contourArea, reverse= True)[:30]
        NumberPlateCnt = None
        img2 = image.copy()
        cv2.drawContours(img2, cnts, -1, (0, 255, 0), 3)

        count = 0
        idx =7
        for c in cnts:
            peri = cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            if len(approx) == 4:
                Traffic_signCnt = approx

                x,y,w,h = cv2.boundingRect(c)
                new_img = image[y:y + h, x:x + w]
                cv2.imwrite("Cropped image containing text" + str(idx) + ".png", new_img)
                idx+= 1
            break



        cv2.drawContours(image, [Traffic_signCnt], -1, (0,255,0), 3)

        crop_img_loc = "Cropped image containing text7.png"
        cv2.imwrite(path, cv2.imread(crop_img_loc))

        # START OF EXTRACTION PROCESS

        # Resize and grayscale the image
        image1 = imutils.resize(cv2.imread(crop_img_loc), width=400)
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        # Find Text-boxes in the image

        h, w, c = image1.shape
        boxes = pytesseract.image_to_boxes(image1)
        for b in boxes.splitlines():
            b = b.split(' ')
            img = cv2.rectangle(image1, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)


        if(image_upload.shape == image_f1.shape or image_upload.shape == image_f2.shape or image_upload.shape == image_f3.shape
                or image_upload.shape == image_s1.shape or image_upload.shape == image_s2.shape or image_upload.shape == image_s3.shape or image_upload.shape == image_s5.shape
            or image_upload.shape == image_f7.shape or image_upload.shape == image_f9.shape):

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
            text2 = re.findall(r'[A-Z]+', text)
            module_dir = os.path.dirname(__file__)  
            file_path = os.path.join(module_dir, 'test.txt')

            with open('test.txt', 'w+') as f:
                for text2 in text2:
                    print(text2, end = ' ')
                    f.write(str( '%s' + ' ' ) % (text2))

                f.seek(0)
                #data = f.read()
                f.closed
            return path
            #file = open('C:/Users/DELL/Dev/BEdjango/src/update_test.txt', 'w')
            

        elif(image_upload.shape == image_f6.shape or image_upload.shape == image_f4.shape or image_upload.shape == image_f5.shape
             or image_upload.shape == image_f8.shape  or image_upload.shape == image_f10.shape or image_upload.shape == image_f11.shape
             or image_upload.shape == image_s4.shape):

            # Removal of noise
            noise = cv2.blur(gray, (3, 3))


            # thresholding the image
            thres = cv2.threshold(noise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


            # opening
            kernel = np.ones((2, 2), np.uint8)
            opening = cv2.morphologyEx(thres, cv2.MORPH_OPEN, kernel)
        

            # Extract the text
            text = pytesseract.image_to_string(opening, lang="eng")
            print(" The Extracted text  is: \n", text)
            print("\n")

            # Remove the numbers
            custom_config = r'-c tessedit_char_blacklist=0123456789 --psm 6'
            text1 = pytesseract.image_to_string(opening, config=custom_config)
            print("The Extracted text excluding numbers: \n", text1)
            print("\n")

            # Remove all special characters and display final text
            text2 = re.findall(r'[A-Z]+', text)
            module_dir = os.path.dirname(__file__)  
            file_path = os.path.join(module_dir, 'test.txt')

            with open('test.txt', 'w+') as f:
                for text2 in text2:
                    print(text2, end = ' ')
                    f.write(str( '%s' + ' ' ) % (text2))

                f.seek(0)
                #data = f.read()
                f.closed
            return path

        else:
            print("different")
            return path
