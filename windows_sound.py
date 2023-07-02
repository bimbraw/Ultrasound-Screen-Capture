import winsound
import numpy as np

import cv2
import pyautogui

# frequency is set to 500Hz
freq = [2200, 1760, 1320, 880, 440]
test_list = [int(i) for i in freq]

# duration is set to 100 milliseconds
dur = 200

image = pyautogui.screenshot(region=(640, 50, 640, 640))
image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
cv2.imwrite("cropped_image.png", image)

print(image.shape)

cv2.imshow('image', image)
cv2.waitKey(0)



'''
#winsound.Beep(freq, dur)
p = 0
q = 0
a = 1
for j in range(0, 5):
     for i in range(0, 500):
         #top left corner to bottom right corner
         if q < 100:
             print(p)
             for pp in range(0, 100):
                  for qq in range(0, 100):
                       a = np.power((pp+1), (qq+1))

         else:
              winsound.Beep(freq[p], dur)
              p = p + 1
              q = 0

         q = q + 1
     winsound.Beep(freq[4], 200)
     p = 0
     q = 0
     a = 1
'''