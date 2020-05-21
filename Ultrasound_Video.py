import cv2
import numpy as np
import os
import pyautogui
from moviepy.editor import VideoFileClip
from moviepy.video.fx.crop import crop

output = "video.avi"
img = pyautogui.screenshot()#region=(640, 0, 640, 800))
img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#get info from img
height, width, channels = img.shape
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

while True:
    img = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imshow('test', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    out.write(image)
    #StopIteration(0.5)

out.release()
cv2.destroyAllWindows()