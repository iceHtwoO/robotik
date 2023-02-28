import cv2
import socket
import numpy
import pickle

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = "192.168.2.2"
port = 6667
s.bind((ip, port))

while True:
    x = s.recvfrom(1000000)
    clientip = x[1][0]
    data = x[0]
    print(data)
    data = pickle.loads(data)
    print(type(data))
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    cv2.imshow('server', data)  # to open image
    if cv2.waitKey(10) == 13:
        breakcv2.destroyAllWindows()
