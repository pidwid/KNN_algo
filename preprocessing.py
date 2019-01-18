## Copyright to rishabh shukla #theboss __ keep away

import cv2;
import numpy as np;

def preprocess(path,name):

    print("Preprocessing Starts!!")
    # Read the Image
    image = cv2.imread(path + name)

    # convert to grayscale
    gray_image = cv2.cvtColor ( image, cv2.COLOR_BGR2GRAY )

    # Dilation and Erosion
    kernel = np.ones ( (2, 2), np.uint8 )
    dilate = cv2.dilate ( gray_image, kernel, iterations=1 )
    erode = cv2.erode ( dilate, kernel, iterations=1 )
    gray_image = erode

    # blur it
    blurred_image = cv2.GaussianBlur ( gray_image, (7, 7), 0 )

    # Otsu's Binarization
    a, binary_image = cv2.threshold ( blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )

    # Run the Canny edge detector
    canny = cv2.Canny ( binary_image, 100, 150 )

    # Converting to bgr
    gray_to_bgr = cv2.cvtColor ( binary_image, cv2.COLOR_GRAY2BGR )

    # converting to hsv
    hsv = cv2.cvtColor ( gray_to_bgr, cv2.COLOR_BGR2HSV )

    # using HSL to mark white
    sensitivity = 80
    lower_white = np.array ( [0, 0, 255 - sensitivity] )
    upper_white = np.array ( [255, sensitivity, 255] )
    # black background pe White text
    mask = cv2.inRange ( hsv, lower_white, upper_white )

    # Converting Black And white
    mask = 255 - mask

    # Writing Masked Image
    cv2.imwrite ( path + "preprocessed.jpg", mask )
    print("Preprocessing Complete!!")

