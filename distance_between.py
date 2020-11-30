# importing the module
import cv2
import math


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
        print(calculateDistance(x, y, 640, 720))

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# driver function
if __name__ == "__main__":
    # reading the image
    img = cv2.imread('./out6/frame2466.png', 1)

    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)
    # print("Distance")
    # print(calculateDistance(750, 442, 640, 720))
    # cv2.line(img, (750, 442), (640, 720), (0, 0, 255), thickness=3, lineType=8)
    # # wait for a key to be pressed to exit
    print("Angle")
    angle = math.atan2(1104 - 720, 441 - 640)
    angleDeg = math.degrees(angle)
    print(abs(angleDeg))
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()
