import cv2, queue as Queue, threading, time
import socket
import pickle
from picarx import Picarx
from robot_hat import TTS
import os
import time
import numpy as np
import atexit
import threading
from flask import Flask
import yaml

config = yaml.safe_load(open("config.yml"))

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 10000000)

px = Picarx()
#tts_robot = TTS()

camera_matrix = np.array([[604.070295286617,0.0,311.3628902968429],[0.0,604.1024918136616,235.21752333703026],[0.0,0.0,1.0]])
dist_coeff = np.array([0.26268929438329897,-1.4747738850911642,-0.001194422721057746,-0.0009405230479656685,2.5718806007625026])

#Bufferless Video
class VideoCaptureQ:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = Queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                global is_frame
                is_frame = False
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except Queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

cap = VideoCaptureQ(0)
config['robot']['video_downscale'] = 2

app = Flask(__name__)

timing = {}

def loop():
    T = 0
    while True:
        start = time.time()
        dist = px.ultrasonic.read() # in CM
        gsd = px.get_grayscale_data() # brighter = higher; first value is right

        img = cap.read()
        global img_out
        img_out = img

        visual(img)

        if cv2.waitKey(10) == 13:
            break
        display_text('FPS:'+str(int(1/T)), (10,80))
        display_text(str(timing),(10,1000))
        display_text(str(timingToPercent()), (10,120))

        start = time.time()
        send_feed_server(img_out)
        timing['send_img'] = time.time() - start

        T = time.time()-start
        timing['total'] = T
    
    cap.release()
    cv2.destroyAllWindows()

def visual(img):
    start = time.time()
    img = undistort_downscale_gray(img)
    timing['img_preparation'] = time.time() - start

    img_width = img.shape[1]
    img_height = img.shape[0]

    line_mask = create_linemask_from_img(img)
    
    display_street_isolation(img)

    start = time.time()
    linesP = find_hough_lines(line_mask)
    timing['find_lines'] = time.time() - start
    
    #lines_with_angle = get_line_and_angle(linesP)
    #left, right = sort_line_by_location(lines_with_angle)
    left, right = sort_line_by_kmeans(linesP)
    display_hughlines_kmeans(left, right)

    bottom_left, top_left = create_boundaryLine_points(left)
    bottom_right, top_right = create_boundaryLine_points(right)

    cv2.circle(img_out, [top_left[0]*config['robot']['video_downscale'],top_left[1]*config['robot']['video_downscale']], 3, (255,255,0), 2)
    cv2.circle(img_out, [top_right[0]*config['robot']['video_downscale'],top_right[1]*config['robot']['video_downscale']], 3, (255,255,0), 2)
    
    display_point_info(bottom_left, top_left, len(left), bottom_right, top_right, len(right))
    display_street_boundaries(bottom_left,top_left,bottom_right,top_right)

    top_center, bottom_center = create_center_points(top_left,top_right,bottom_left,bottom_right)
    #bottom_center = [int(img.shape[1]/2),img.shape[0]] #Fixed bottom
    center_vector = [top_center[0]-bottom_center[0], top_center[1]-bottom_center[1]]

    display_street_center(top_center,bottom_center)

    steer_to_center(center_vector,img_width/2-bottom_center[0])
    
    display_img_mid()

    ret, thresh = cv2.threshold(line_mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_out, contours, -1, (0,255,0), 3)

def undistort_downscale_gray(img):
    #img = cv2.undistort(img,camera_matrix, dist_coeff,None,camera_matrix)
    img = img[20:img.shape[2]-20,20:img.shape[1]-20]
    img = cv2.resize(img,(int(img.shape[1] * 1/config['robot']['video_downscale']),int(img.shape[0] * 1/config['robot']['video_downscale'])))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def steer_to_center(center_vector,delta_bottomx):
    offset_angle = rad_to_degree(get_angle_between_vectors_with_direction(center_vector,[0,-1]))
    angle = street_offset_to_steer_angle(offset_angle)
    display_steer_info(angle)
    px.set_dir_servo_angle(angle)

def street_offset_to_steer_angle(x):
    angle = -x/45 * config['robot']['laneDetection']['maxSteer']
    angle = np.clip(angle, -config['robot']['laneDetection']['maxSteer'], config['robot']['laneDetection']['maxSteer'])
    return angle

def create_center_points(top1,top2, bottom1, bottom2):
    return [int((top1[0]+top2[0])/2), int((top1[1]+top2[1])/2)], [int((bottom1[0]+bottom2[0])/2),int((bottom1[1]+bottom2[1])/2)]

def create_linemask_from_img(img):
    line_mask = cv2.inRange(img, np.array([0], dtype = "uint8"),np.array(config['robot']['laneDetection']['brightness'], dtype = "uint8"))
    line_mask = cv2.Canny(line_mask,20,150)
    line_mask = cv2.GaussianBlur(line_mask, (5,5), 0)
    return isolate_street(line_mask)

def isolate_street(img):
    height, width = img.shape
    triangle = np.array([[(0, height),(0, int(height*0.75)), (int(width/2)-int(width*0.40), int(height/2)), (int(width/2)+int(width*0.40), int(height/2)),(width, int(height*0.75)), (width, height)]])
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, triangle, color=(255, 0, 0))
    mask = cv2.bitwise_and(img, mask)
    return mask

def find_hough_lines(img):
    return cv2.HoughLinesP(img,1,np.pi/180,30)

def cluster_line_by_kmeans(lines):
    samples = np.float32([[i.reshape(4)[0],i.reshape(4)[1]] for i in lines])
    k = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(samples,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    return lines[label.ravel()==0], lines[label.ravel()==1],center

def sort_line_by_kmeans(lines):
    if lines is not None and len(lines)>1:
        l1,l2,centers = cluster_line_by_kmeans(lines)
        return (l1,l2) if centers[0][0]<centers[1][0] else (l2,l1)
    else:
        return [],[]

def create_boundaryLine_points(line_array):
    return find_heighest_y_point(line_array),find_lowest_y_point(line_array)

def find_lowest_y_point(line_array):
    lowest_y = [0,10000000]
    for line in line_array:
        x1, y1, x2, y2 = line.reshape(4)
        if y1 < lowest_y[1]:
            lowest_y = [x1,y1]
        elif y2 < lowest_y[1]:
            lowest_y = [x2,y2]
    return lowest_y

def find_heighest_y_point(line_array):
    heighest = [0,0]
    for line in line_array:
        x1, y1, x2, y2 = line.reshape(4)
        if y1 > heighest[1]:
            heighest = [x1,y1]
        elif y2 > heighest[1]:
            heighest = [x2,y2]
    return heighest

def display_point_info(bottom_left, top_left,leftlen, bottom_right, top_right,rightlen):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.75
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2
    cv2.putText(img_out,'UL:'+str(top_left)+" UR:"+str(top_right)+" Count:"+str(leftlen), 
    (10,20), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    cv2.putText(img_out,'BL:'+str(bottom_left)+" BR:"+str(bottom_right)+" Count:"+str(rightlen), 
    (10,40), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)

def display_img_mid():
    cv2.line(img_out,(int(img_out.shape[1]/2),0),(int(img_out.shape[1]/2),img_out.shape[0]),(0,255,0),1)

def display_street_center(top_center,bottom_center):
    top_center = upscale_cords_to_original_img(top_center)
    bottom_center = upscale_cords_to_original_img(bottom_center)
    try:
        cv2.line(img_out,top_center,bottom_center,(255,0,255),10)
        cv2.circle(img_out, bottom_center, 3, (255,255,0), 2)
        cv2.circle(img_out, top_center, 3, (255,0,0), 2)
    except:
        print("Err while drawing lines")

def display_street_boundaries(bottom_left,top_left,bottom_right,top_right):
    try:
        cv2.line(img_out,upscale_cords_to_original_img(bottom_left),upscale_cords_to_original_img(top_left),(0,255,0),2)
        cv2.line(img_out,upscale_cords_to_original_img(bottom_right),upscale_cords_to_original_img(top_right),(0,255,0),2)
    except:
        print("Err while drawing lines")

def display_hughlines(lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(img_out,upscale_cords_to_original_img([x1,y1]),upscale_cords_to_original_img([x2,y2]),(255,0,0),4)

def display_hughlines_kmeans(linesA, linesB):
    if linesA is not None:
        for line in linesA:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(img_out,upscale_cords_to_original_img([x1,y1]),upscale_cords_to_original_img([x2,y2]),(255,0,0),4)
    if linesB is not None:
        for line in linesB:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(img_out,upscale_cords_to_original_img([x1,y1]),upscale_cords_to_original_img([x2,y2]),(0,0,255),4)

def upscale_cords_to_original_img(cords):
    return [config['robot']['video_downscale']*cords[0],config['robot']['video_downscale']*cords[1]]

def display_street_isolation(img):
    uwidth = img.shape[1]*config['robot']['video_downscale']
    uheight = img.shape[0]*config['robot']['video_downscale']
    triangle = np.array([[(0, uheight),(0, int(uheight*0.75)), (int(uwidth/2)-int(uwidth*0.40), int(uheight/2)), (int(uwidth/2)+int(uwidth*0.40), int(uheight/2)),(uwidth, int(uheight*0.75)), (uwidth, uheight)]])
    cv2.polylines(img_out,triangle,True,(0,255,255))

def display_steer_info(angle):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.75
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2
    cv2.putText(img_out,'Steer Angle:'+str(angle)+"", 
    (10,60), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)

def display_text(text,cords):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.75
    fontColor              = (255,255,255)
    thickness              = 1
    lineType               = 2
    cv2.putText(img_out,text, 
    cords, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)

def transform_image(img,bottom_left,bottom_right,top_left,top_right):
    pts1 = np.float32([bottom_left,bottom_right,top_left,top_right])
    pts2 = np.float32([[0, img.shape[0]], [img.shape[1], img.shape[0]],[0, 0], [img.shape[1], 0]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (700, 400))

def get_angle_between_vectors(vector1,vector2):
    return np.arccos(np.dot(vector1,vector2) / np.sqrt(vector1[0]**2 + vector1[1]**2) * np.sqrt(vector2[0]**2 + vector2[1]**2))

def get_angle_between_vectors_with_direction(vector1,vector2):
    return np.arctan2(vector1[0]*vector2[1]-vector1[1]*vector2[0],np.dot(vector1,vector2))

def rad_to_degree(rad):
    return (rad / np.pi) * 180

def servo_check():
    time.sleep(5)
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

def send_feed_server(photo):
    ret, buffer = cv2.imencode(
        ".jpg", photo, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    x_as_bytes = pickle.dumps(buffer)

    s.sendto(x_as_bytes, (config['robot']['server']['ip'], config['robot']['server']['port']))

def exit_handler():
     px.forward(0)

def timingToPercent():
    timing_percent = {}
    for key, value in timing.items():
        if key is 'total':
            continue
        else:
            timing_percent[key] = int(value/timing["total"] * 100)
    return timing_percent

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == "__main__":
    if config['robot']['servoCheck']:
        servo_check()
    atexit.register(exit_handler)
    px.set_camera_servo1_angle(0)
    if config['robot']['move']:
        px.forward(1)
    px.set_camera_servo2_angle(-11)
    loop()