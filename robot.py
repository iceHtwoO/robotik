import cv2
import socket
import pickle
from picarx import Picarx
from robot_hat import TTS
import os
import time


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 10000000)

serverip = "192.168.2.2"
serverport = 6666

px = Picarx()
tts_robot = TTS()

cap = cv2.VideoCapture(0)
camera_matrix = np.array([[604.070295286617,0.0,311.3628902968429],[0.0,604.1024918136616,235.21752333703026],[0.0,0.0,1.0]])
dist_coeff = np.array([0.26268929438329897,-1.4747738850911642,-0.001194422721057746,-0.0009405230479656685,2.5718806007625026])

def loop():
    while True:
        dist = px.ultrasonic.read() # in CM
        gsd = px.get_grayscale_data() # brighter = higher; first value is right
        print(gsd)

        ret, img = cap.read()
    
        img = cv2.undistort(img,camera_matrix, dist_coeff,None,camera_matrix)
        send_feed_server(img)

        if cv2.waitKey(10) == 13:
            break
        cv2.destroyAllWindows()
    cap.release()

def servo_check():
    px.set_dir_servo_angle(-30)
    time.sleep(1)
    px.set_dir_servo_angle(30)
    time.sleep(1)
    px.set_dir_servo_angle(0)
    time.sleep(1)

    px.set_camera_servo1_angle(-30)
    px.set_camera_servo2_angle(-30)
    time.sleep(1)
    px.set_camera_servo1_angle(90)
    px.set_camera_servo2_angle(90)
    time.sleep(1)
    px.set_camera_servo1_angle(0)
    px.set_camera_servo2_angle(-10)
    time.sleep(1)

def send_feed_server(photo):
    ret, buffer = cv2.imencode(
        ".jpg", photo, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    x_as_bytes = pickle.dumps(buffer)

    s.sendto(x_as_bytes, (serverip, serverport))


if __name__ == "__main__":
    servo_check()
    loop()