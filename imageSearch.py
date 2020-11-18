# import necessary packages
from imutils import paths
import numpy as np
import decimal
import imutils
import cv2


def find_marker(image):
    # convert the image to grayscale, blur it and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)

    # find contours and keep largest
    # assume this is boat
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # computer bounding box and return (1060, 446, 90, 54)  (884, 433, 40, 14) (678, 432, 76, 36) ((716, 450),(76,36), -0.0)
    #((45, 446),(90, 54), -0.0) ((1050, 433),(40, 14), -0.0)  (1091, 425, 28, 16) ((1105,433), (28, 16), -0.0)
    return ((716, 450),(76,36), -0.0)#((1105,433), (28, 16), -0.0)#((902, 437),(40, 14), -0.0) #cv2.minAreaRect(c)


def distance_to_camera(knownWidth, focalLength, perWidth):
    # computer and return distance from the marker to the camera
    if 50 < perWidth < 70:
        perWidth = perWidth + 26
    elif 20 < perWidth < 50:
        perWidth = perWidth + 18
    return (knownWidth * focalLength) / perWidth


KNOWN_DISTANCE = 2789.7638

KNOWN_WIDTH = 192

image = cv2.imread("out6/frame3000.png")
# bbox = cv2.selectROI(image, False)
# print(bbox)

marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH


for imagePath in sorted(paths.list_images("test")):
    image = cv2.imread(imagePath)
    marker = find_marker(image)
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
    #
    #  frame 1848: (243, 426, 57, 33) ((271.5,442.5),(57,33),-0.0)
    # ((273,444),(54, 28),-0.0)
    # frame 257: (1092, 424, 28, 16) ((1106,432),(28,16),-0.0)
    # frame 3000: (741, 426, 25, 15) ((753.5,439.5)(25,15),-0.0)
    marker = ((1106, 432), (28, 16), -0.0)
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0]) / 12
    print("%.2fft" % inches)
    cv2.waitKey(0)
