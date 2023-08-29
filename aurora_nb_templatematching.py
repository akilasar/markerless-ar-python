import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from ROI import ROI
from Matcher import Matcher
import argparse
import imutils
import yaml
from ultralytics import YOLO
from math import pi,atan2,asin
from scipy.spatial.transform import Rotation as R

with open("../calibration_boards/calibration_matrix.yaml", "r") as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

cameraMatrix = np.array(data['camera_matrix'])
disCoeff = np.array(data['dist_coeff'])

def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))
    return writer

def mask_from_contours(ref_img, contours):
    mask = np.zeros(ref_img.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255,255,255), thickness=cv2.FILLED)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

def segment_book(img, counter):
    image = img
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw_copy = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    blur = cv2.bilateralFilter(gray, 5, 75, 75)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    mask = np.zeros(closed.shape, dtype=np.uint8)
    mask1 = np.zeros(bw_copy.shape, dtype=np.uint8)
    wb_copy = cv2.bitwise_not(bw_copy)
    new_bw = np.zeros(bw_copy.shape, dtype=np.uint8)

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y + h, x:x + w] = 0
        area = cv2.contourArea(contours[idx])
        aspect_ratio = float(w) / h
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)

        if r > 0.34 and 0.5 < aspect_ratio < 13 and area > 500.0:

            cv2.drawContours(mask1, [contours[idx]], -1, (255, 255, 255), -1)
            bw_temp = cv2.bitwise_and(mask1[y:y + h, x:x + w],bw_copy[y:y + h, x:x + w])
            wb_temp = cv2.bitwise_and(mask1[y:y + h, x:x + w],wb_copy[y:y + h, x:x + w])

            bw_count = cv2.countNonZero(bw_temp)
            wb_count = cv2.countNonZero(wb_temp)

            if bw_count > wb_count:
                new_bw[y:y + h, x:x + w]=np.copy(bw_copy[y:y + h, x:x + w])
            else:
                new_bw[y:y + h, x:x + w]=np.copy(wb_copy[y:y + h, x:x + w])
    (y, x) = np.where(new_bw == 255)
    if y.shape != (0,) and x.shape != (0,):
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))
        out = image[topy:bottomy+1, topx:bottomx+1]
        out_mask = new_bw[topy:bottomy+1, topx:bottomx+1]
        out_coords = (topx, topy, bottomx, bottomy)
    else:
        out = None
        out_mask = None
        out_coords = tuple([None]*4)
    return out, out_mask, out_coords

class App:
    def __init__(self, alg, mode):
        self.referencePoints = []
        self.cropping = False
        self.roi = None # the marker object
        self.alg = alg # the algorithm to choose: sift/orb
        self.mode = mode # the way of choosing marker: static/screen capture

    def click_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.referencePoints = [(x, y)]
            self.cropping = True
     
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            self.referencePoints.append((x, y))
            self.cropping = False

    def draw(self, img, corners, imgpts, angle, center):  #simple pose axes 
        corners = [corn + center for corn in corners] #np.array([img.shape[1]//2, img.shape[0]//2])
        corner = tuple(corners[0].ravel().astype(int)[:2]) #+ np.array([img.shape[1]//2, img.shape[0]//2]))
        img = cv2.polylines(img,corners,True,(255,255,255),2, cv2.LINE_AA)
        qx0 = corner[0] + np.cos(np.deg2rad(angle - 90))*100 #- np.sin(np.deg2rad(angle))*5
        qy0 = corner[1] + np.sin(np.deg2rad(angle - 90))*100 #angle #corner[1] + np.sin(np.deg2rad(angle))*(tuple(imgpts[0].ravel().astype(int))[0] - corner[0]) + np.cos(np.deg2rad(angle))*(tuple(imgpts[0].ravel().astype(int))[1] - corner[1])

        # qx1 = corner[0] + np.cos(np.deg2rad(angle))*100 #- np.sin(np.deg2rad(angle))*5
        # qy1 = corner[1] + np.sin(np.deg2rad(angle))*100 #angle #corner[1] + np.sin(np.deg2rad(angle))*(tuple(imgpts[0].ravel().astype(int))[0] - corner[0]) + np.cos(np.deg2rad(angle))*(tuple(imgpts[0].ravel().astype(int))[1] - corner[1])

        img = cv2.line(img, corner, tuple([int(qx0), int(qy0)]), (255,0,0), 5) #blue
        #img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int) + center), (0,255,0), 5) #green
        #img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int) + center), (0,0,255), 5) #red
        return img

    def main(self):
        cap = cv2.VideoCapture('IMG_8554.mov') #use 0 for webcam
        writer = create_video_writer(cap, "out_sift_m6_book.mp4")
        counter = 0
        isEnough = True
        
        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        heading = 178.538983356
        prev_euler = 0
        rollover_euler = 0

        old_book = None
        old_mask = None
        old_coords = []

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            if(counter%1 == 0):
                image = frame.copy()
                book, book_mask, coords = segment_book(image, counter)
                if book is None:
                    book = old_book
                    book_mask = old_mask
                    coords = old_coords

                if counter == 0:
                    self.roi = ROI(book, self.alg)
                    matcher = Matcher(self.roi, self.alg, disCoeff, cameraMatrix)
                    old_corners = self.roi.getPoints3d()
                    #prev_euler = ang

                currentFrame = frame.copy()
        
                cv2.namedWindow('webcam')
                matcher.setFrame(book)
                
                result = matcher.getCorrespondence()
                if result:
                    (src, dst, corners, M) = result
                else:
                    print('Not enough points')
                    self.roi = ROI(book, self.alg)
                    matcher = Matcher(self.roi, self.alg, disCoeff, cameraMatrix)
                    rollover_euler = prev_euler
                    isEnough = False
                    continue
                
                (retvalCorner, rvecCorner, tvecCorner) = matcher.computePose(self.roi.getPoints3d(), corners)
                if counter == 0:
                    prev_euler =  (R.from_rotvec(rvecCorner.reshape(3,))).as_euler('zxy', degrees=True)[0]
                    euler = prev_euler
                else:
                    if not isEnough:
                        euler = rollover_euler
                        prev_euler =  (R.from_rotvec(rvecCorner.reshape(3,))).as_euler('zxy', degrees=True)[0]
                        isEnough = True
                    else:
                        euler = (R.from_rotvec(rvecCorner.reshape(3,))).as_euler('zxy', degrees=True)[0]
                print(euler)
                delta = euler - prev_euler
                heading -= delta
                prev_euler = euler
                if retvalCorner:
                    axis = np.float32([[0.5,0,0], [0,0.5,0], [0,0,0.5]]).reshape(-1,3)
                    imgpts, jac = cv2.projectPoints(axis, rvecCorner, tvecCorner, cameraMatrix, disCoeff)

                    # re-draw the frame
                    currentFrame = self.draw(currentFrame, [np.int32(corners)], imgpts, heading, np.array([coords[0],coords[1]])) 
                    print(currentFrame.shape)
                    fps = "Heading: " + str(int((heading + 180)%360))
                    cv2.putText(currentFrame, fps, (50, 50),
                      cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
                    #import ipdb; ipdb.set_trace()
                    resizeframe = imutils.resize(currentFrame, width=500)
                    writer.write(currentFrame)
                    cv2.imshow('webcam', resizeframe)
                    cv2.waitKey(1)
                    
                else:
                    cv2.waitKey(1)
                    continue

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    writer.release()
                    cv2.destroyAllWindows()
                    break
            counter += 1
            old_book = book
            old_mask = book_mask
            old_coords = coords
            print(counter)

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = App("cnn", "static")
    app.main()