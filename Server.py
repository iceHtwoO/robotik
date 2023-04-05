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
        cv2.imshow('server', data)  # to open image
        #show_data(data)
        if cv2.waitKey(10) == 13:
            breakcv2.destroyAllWindows()

def show_data(img):
    img = img[10:img.shape[2]-10,10:img.shape[1]-10]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

    try:
        cv2.line(img,p1l,p2l,(0,255,0),2)
        cv2.line(img,p1r,p2r,(0,255,0),2)
        cv2.line(img,(int((p2l[0]+p2r[0])/2),p2l[1]),(int((p1l[0]+p1r[0])/2),p1l[1]),(255,0,255),10)
    except:
        print("line Not found")
    cv2.line(img,(int(img.shape[1]/2),0),(int(img.shape[1]/2),img.shape[0]),(0,255,0),1)

    distance_to_center = img.shape[1]/2 - int((p2l[0]+p2r[0])/2)

    print("DELTA")
    print(distance_to_center)

    #cv2.line(img,((p1r[0]-p1l[0]),p1r[1]),((p2r[0]-p2l[0]),p2r[1]),(0,0,255),10)

    #print(p1l)


    #display_lines(img, np.array([make_points(img, left_avg), make_points(img, right_avg)]))

    im1.set_data(img)
    im2.set_data(cv2.cvtColor(line_mask, cv2.COLOR_BGR2RGB))
    im3.set_data(transform_image(img,p1l,p1r,p2l,p2r))
    plt.pause(0.01)


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
        elif line["angle"] < -20:
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

loop()