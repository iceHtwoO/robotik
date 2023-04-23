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

ax1 = plt.subplot(1,3,1)
ax2 = plt.subplot(1,3,2)
ax3 = plt.subplot(1,3,3)
plt.subplots_adjust(wspace=0, hspace=0)
plt.ion()

im1 = ax1.imshow(cv2.imread('.\placeholder.jpg'))
im2 = ax2.imshow(cv2.imread('.\placeholder.jpg'))#cmap='gray'
im3 = ax3.imshow(cv2.imread('.\placeholder.jpg'))

def loop():
    while True:
        x = s.recvfrom(1000000)
        clientip = x[1][0]
        data = x[0]
        data = pickle.loads(data)
        data = cv2.imdecode(data, cv2.IMREAD_COLOR)
        data = cv2.resize(data,(int(data.shape[1] * 2.5),int(data.shape[0] * 2)))
        cv2.imshow('server', data)  # to open image
        if cv2.waitKey(10) == 13:
            cv2.destroyAllWindows()
loop()