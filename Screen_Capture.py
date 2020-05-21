#Author - Keshav Bimbraw

import numpy as np
import cv2
import pyautogui
from screeninfo import get_monitors
import time

#With a 0.1 s sleep - the rate of data collection
#was around 5 Hz. For no sleep - around 14 Hz
time_start = time.perf_counter()

#This gives the monitors data - can be used to fix the region parameters
print(get_monitors())

for i in range(0, 100):
    #top left corner to bottom right corner
    image = pyautogui.screenshot(region=(640, 0, 640, 800))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite("image" + str(i) + ".png", image)
    #Uncomment it to space out the screenshots wrt time
    #time.sleep(0.1)

time_end = time.perf_counter()