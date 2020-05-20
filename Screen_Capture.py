import numpy as np
import cv2
import pyautogui
from screeninfo import get_monitors
import time

print(get_monitors())

for i in range(0, 100):
    image = pyautogui.screenshot(region=(640, 0, 640, 800))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite("image" + str(i) + ".png", image)
    time.sleep(0.1)
