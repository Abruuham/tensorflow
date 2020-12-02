# import necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2


#boat at frame 1400
KNOWN_DISTANCE = 2789.7638

#estimate of the length of the boat in inches
KNOWN_WIDTH = 192

# reading image
image = cv2.imread("frame3000.png")

# this is just to make a bounding box
# bbox = cv2.selectROI(image, False)
# print(bbox)

#minAreaRect of the boat at frame 1400, used for calibration
marker = ((716, 450),(76,26), -0.0)

# getting the first number from the second set of the paranthesis ((,)(x,)())
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH



def distance_to_camera(knownWidth, focalLength, perWidth):
    # computer and return distance from the marker to the camera
    if 50 < perWidth <= 55:
        perWidth = perWidth + 26
    elif 55 < perWidth <= 60:
        perWidth = perWidth + 23
    elif 25 < perWidth <= 30:
        perWidth = perWidth + 18.1
    elif 20 < perWidth <= 25:
        perWidth = perWidth + 13.5
    return (knownWidth * focalLength) / perWidth




for imagePath in sorted(paths.list_images("test")):

    ''' this is to display the known distance of the boat in feet'''
    image = cv2.imread(imagePath)
    marker = ((716, 450),(76,26), -0.0)
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    inches = inches / 12
    box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
    box = np.int0(box)
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    cv2.putText(image, "%.2fft" % (inches),
                (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (0, 255, 0), 3)
    cv2.imshow("image", image)
    print(inches)




    # test cases with bounding boxes and minAreaRects
    #  frame 1848: (243, 426, 57, 33) ((271.5,442.5),(57,33),-0.0)
    # ((273,444),(54, 28),-0.0)
    # frame 257: (1092, 424, 28, 16) ((1106,432),(28,16),-0.0)
    # frame 3000: (741, 426, 25, 15) ((753.5,439.5),(25,15),-0.0)
    marker = ((1106,432),(28,16),-0.0)
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0]) / 12
    print("%.2fft" % inches)
    cv2.waitKey(0)
