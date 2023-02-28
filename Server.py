import cv2
import socket
import numpy as np
import pickle
from matplotlib import pyplot as plt
import time
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = "192.168.2.2"
port = 6666
s.bind((ip, port))

ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
plt.ion()

im1 = ax1.imshow(cv2.imread('.\placeholder.jpg'))
im2 = ax2.imshow(cv2.imread('.\placeholder.jpg'))#cmap='gray'

def loop():
    while True:
        x = s.recvfrom(1000000)
        clientip = x[1][0]
        data = x[0]
        print(data)
        data = pickle.loads(data)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        #cv2.imshow('server', data)  # to open image
        show_data(data)
        if cv2.waitKey(10) == 13:
            breakcv2.destroyAllWindows()

def isolate_street(img):
    height, width = img.shape
    triangle = np.array([[(0, height),(0, height-70), (int(width/2)-100, int(height/2)), (int(width/2)+100, int(height/2)),(width, height-70), (width, height)]])
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, triangle, color=(255, 0, 0))
    mask = cv2.bitwise_and(img, mask)
    return mask

def get_lines(img):
    return cv2.HoughLinesP(img,1,np.pi/180,10, 40, 5)

def show_data(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    line_mask = cv2.inRange(img, np.array([0,0,0], dtype = "uint8"),np.array([100,100,100], dtype = "uint8"))
    line_mask = cv2.Canny(line_mask,20,150)
    line_mask = isolate_street(line_mask)

    linesP = get_lines(line_mask)
    if linesP is not None:
        for line in linesP:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),10)

    im1.set_data(img)
    im2.set_data(cv2.cvtColor(line_mask, cv2.COLOR_BGR2RGB))
    plt.pause(0.05)

loop()