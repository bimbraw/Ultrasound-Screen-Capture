#Author - Keshav Bimbraw

import numpy as np
import cv2
import pyautogui
from screeninfo import get_monitors
import time
import winsound

#With a 0.1 s sleep - the rate of data collection
#was around 5 Hz. For no sleep - around 14 Hz
time_start = time.perf_counter()

# frequency is set to 500Hz
freq = [587, 659, 699, 784, 523]
test_list = [int(i) for i in freq]

# duration is set to 100 milliseconds
dur = 200

rounds = 6
classes = 5
len_classes = 100
configurations = ["Perpendicular_1", "Perpendicular_2", "Perpendicular_3", "Parallel_upwards", "Parallel_downwards"]
configuration = configurations[0]
subjects = ["Keshav", "Anthony", "Camren", "Layal", "Kevin"]
subject = subjects[3]

winsound.Beep(freq[4], 200)
p = 0
q = 0
a = 1
for j in range(0, rounds):
     for i in range(0, classes*len_classes):
         image = pyautogui.screenshot(region=(640, 50, 640, 640))
         image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
         cv2.imwrite("P:/MQP_data/" + str(configuration) + "/" + str(subject) + "/image" + str(j) + str(i) + ".png", image)
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

time_end = time.perf_counter()
print(time_end)

frame_rate = (classes*len_classes*rounds)/time_end
text_file = open("P:/MQP_data/" + str(configuration) + "/" + str(subject) + "/frame_rate.txt", "w")
n = text_file.write(str(frame_rate) + " Hz")
text_file.close()