import cv2

class Display:
    img_out = []
    scale = 1
    def __init__(self, img, scale):
        self.img_out = img
        self.scale = scale

    def set_image(self, img):
        self.img_out = img

    def get_img(self):
        return self.img_out

    def display_point_info(self, bottom_left, top_left,leftlen, bottom_right, top_right,rightlen):
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.75
        fontColor              = (255,255,255)
        thickness              = 1
        lineType               = 2
        cv2.putText(self.img_out,'UL:'+str(top_left)+" UR:"+str(top_right)+" Count:"+str(leftlen), 
        (10,20), 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)
        cv2.putText(self.img_out,'BL:'+str(bottom_left)+" BR:"+str(bottom_right)+" Count:"+str(rightlen), 
        (10,40), 
        font, 
        fontScale,
        fontColor,
        thickness,
        lineType)

    def display_img_mid(self):
        cv2.line(self.img_out,(int(self.img_out.shape[1]/2),0),(int(self.img_out.shape[1]/2),self.img_out.shape[0]),(0,255,0),1)

    def display_street_center(self, top_center,bottom_center):
        top_center = upscale_cords_to_original_img(top_center)
        bottom_center = upscale_cords_to_original_img(bottom_center)
        try:
            cv2.line(img_out,top_center,bottom_center,(255,0,255),10)
            cv2.circle(img_out, bottom_center, 3, (255,255,0), 2)
            cv2.circle(img_out, top_center, 3, (255,0,0), 2)
        except:
            print("Err while drawing lines")

    def display_street_boundaries(self, bottom_left,top_left,bottom_right,top_right):
        try:
            cv2.line(img_out,self.upscale_cords_to_original_img(bottom_left),self.upscale_cords_to_original_img(top_left),(0,255,0),2)
            cv2.line(img_out,self.upscale_cords_to_original_img(bottom_right),self.upscale_cords_to_original_img(top_right),(0,255,0),2)
        except:
            print("Err while drawing lines")

    def display_hughlines(self, lines):
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(self.img_out,self.upscale_cords_to_original_img([x1,y1]),self.upscale_cords_to_original_img([x2,y2]),(255,0,0),4)

    def display_hughlines_kmeans(self, linesA, linesB):
        if linesA is not None:
            for line in linesA:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(img_out,self.upscale_cords_to_original_img([x1,y1]),self.upscale_cords_to_original_img([x2,y2]),(255,0,0),4)
        if linesB is not None:
            for line in linesB:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(img_out,self.upscale_cords_to_original_img([x1,y1]),self.upscale_cords_to_original_img([x2,y2]),(0,0,255),4)

    def display_street_isolation(self, width, height):
        uwidth = width*self.scale
        uheight = height*self.scale
        triangle = np.array([[(0, uheight),(0, int(uheight*0.75)), (int(uwidth/2)-int(uwidth*0.40), int(uheight/2)), (int(uwidth/2)+int(uwidth*0.40), int(uheight/2)),(uwidth, int(uheight*0.75)), (uwidth, uheight)]])
        cv2.polylines(img_out,triangle,True,(0,255,255))

    def upscale_cords_to_original_img(self, cords):
        return [self.scale*cords[0],self.scale*cords[1]]