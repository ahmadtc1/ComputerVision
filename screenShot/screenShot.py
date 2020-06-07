import pyautogui as pg
import cv2
import numpy as np
import time
from datetime import datetime

screen = pg.screenshot()
screen = np.array(screen)
screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

#Create the filename with a timestamp
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
fileName = "screenshot@" + current_time + ".jpg"
seperated = fileName.split(sep=":")
period = '.'
fileName = period.join(seperated)

#Save the screenshot with the new fileName
cv2.imwrite(fileName, screen)
print("Successfully saved screenshot to desktop")