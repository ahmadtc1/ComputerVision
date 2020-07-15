import cv2
import numpy as np

rectangle = False # Default to drawing circles, switch to rectangle is triggered
drawing = False
startX, startY = -1, -1
canvas = np.zeros((512,512,3), np.uint8)

def drawShape(event, x, y, flags, param):
    global canvas, startX, startY, drawing, rectangle

    if (event == cv2.EVENT_LBUTTONDOWN):
        startX, startY = x, y
        drawing = True

    elif (event == cv2.EVENT_MOUSEMOVE):
        if (drawing == True):
            if (rectangle == True):
                cv2.rectangle(canvas, (startX, startY), (x, y), (255, 0, 0), -1)

            else:
                center = ((startX + x) // 2, (startY + y) // 2)
                cv2.circle(canvas, center, (x - center[0]), (0, 255, 0), -1)

    elif (event == cv2.EVENT_LBUTTONUP):
        if (rectangle == True):
            cv2.rectangle(canvas, (startX, startY), (x, y), (255, 0, 0), -1)

        else:
            center = ((startX + x) // 2, (startY + y) // 2)
            cv2.circle(canvas, center, (x - center[0]), (0, 255, 0), -1)
        
        drawing = False

cv2.namedWindow('image')
cv2.setMouseCallback('image',drawShape)
while(True):
    cv2.imshow('image',canvas)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        rectangle = not rectangle
    elif k == ord('q'):
        break
cv2.destroyAllWindows()
