import cv2
import socket
import pickle
from picarx import Picarx
from robot_hat import TTS
import os
import time
import numpy as np


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
        #send_feed_server(img)
        visual(img)

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


def visual(img):
    img = img[10:img.shape[2]-10,10:img.shape[1]-10]
    line_mask = cv2.inRange(img, np.array([0,0,0], dtype = "uint8"),np.array([50,50,50], dtype = "uint8"))
    line_mask = cv2.Canny(line_mask,20,150)
    line_mask = isolate_street(line_mask)

    linesP = get_lines(line_mask)
    if linesP is not None:
        for line in linesP:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
    
    lines_with_angle = get_line_and_angle(linesP)
    left, right = sort_line_by_angle(lines_with_angle, img.shape[1])

    p1l, p2l = create_points_left(left)
    p1r, p2r = create_points_right(right)

    center_vector = [int((p2l[0]+p2r[0])/2)-int((p1l[0]+p1r[0])/2), p2l[1]-p1l[1]]
    print(center_vector)

    try:
        cv2.line(img,p1l,p2l,(0,255,0),2)
        cv2.line(img,p1r,p2r,(0,255,0),2)
        cv2.line(img,(int((p2l[0]+p2r[0])/2),p2l[1]),(int((p1l[0]+p1r[0])/2),p1l[1]),(255,0,255),10)
        print(get_offset_angle(center_vector))
    except:
        print("line Not found")
    cv2.line(img,(int(img.shape[1]/2),0),(int(img.shape[1]/2),img.shape[0]),(0,255,0),1)

    #distance_to_center = img.shape[1]/2 - int((p2l[0]+p2r[0])/2)

    send_feed_server(img)

def isolate_street(img):
    height, width = img.shape
    triangle = np.array([[(0, height),(0, height-70), (int(width/2)-100, int(height/2)), (int(width/2)+100, int(height/2)),(width, height-70), (width, height)]])
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, triangle, color=(255, 0, 0))
    mask = cv2.bitwise_and(img, mask)
    return mask

def get_lines(img):
    return cv2.HoughLinesP(img,1,np.pi/180,30, 30)

def rad_to_degree(rad):
    return (rad / np.pi) * 180

def get_line_and_angle(lines):
    lines_out = []
    if lines is not None:
        for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                angle = np.arctan2(y2-y1, x2-x1)
                lines_out.append({"line":line, "angle":rad_to_degree(angle)})
    return lines_out

def sort_line_by_angle(lines, width):
    left = []
    right = []
    for line in lines:
        if line["angle"] > 20 and line["line"].reshape(4)[0]>width/2:
            right.append(line)
        elif line["angle"] < -20 and line["line"].reshape(4)[0]<width/2:
            left.append(line)
    return left, right

def avg_lines(line_array):
    return np.average(line_array)


def average_angle(line_array):
    return np.average([i["angle"] for i in line_array])

def create_points_left(line_array):
    lx,hx,ly,hy = find_extreme_from_lines_array(line_array)
    return (lx,hy),(hx,ly)

def create_points_right(line_array):
    lx,hx,ly,hy = find_extreme_from_lines_array(line_array)
    return (hx,hy),(lx,ly)


def find_extreme_from_lines_array(line_array):
    return find_lowest_x(line_array), find_heighest_x(line_array),find_lowest_y(line_array), find_highest_y(line_array)

def find_lowest_y(line_array):
    lowest_y = 10000000000
    for line in line_array:
        _, y1, _, y2 = line["line"].reshape(4)
        if y1 < lowest_y:
            lowest_y = y1
        elif y2 < lowest_y:
            lowest_y = y2
    return lowest_y

def find_highest_y(line_array):
    heighest_y = 0
    for line in line_array:
        _, y1, _, y2 = line["line"].reshape(4)
        if y1 > heighest_y:
            heighest_y = y1
        elif y2 > heighest_y:
            heighest_y = y2
    return heighest_y

def find_lowest_x(line_array):
    lowest_x = 10000000000
    for line in line_array:
        x1, _, x2, _ = line["line"].reshape(4)
        if x1 < lowest_x:
            lowest_x = x1
        elif x2 < lowest_x:
            lowest_x = x2
    return lowest_x

def find_heighest_x(line_array):
    heighest_x = 0
    for line in line_array:
        x1, _, x2, _ = line["line"].reshape(4)
        if x1 > heighest_x:
            heighest_x = x1
        elif x2 > heighest_x:
            heighest_x = x2
    return heighest_x

def transform_image(img,p1l,p1r,p2l,p2r):
    pts1 = np.float32([p1l,p1r,p2l,p2r])
    pts2 = np.float32([[0, img.shape[0]], [img.shape[1], img.shape[0]],[0, 0], [img.shape[1], 0]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (700, 400))

def get_offset_angle(street_mid):
    return np.arccos(np.dot(street_mid,[0,-1]) / np.sqrt(street_mid[0]**2 + street_mid[1]**2))

if __name__ == "__main__":
    #servo_check()
    loop()