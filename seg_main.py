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
import os

with open("../utils/calibration_boards/calibration_matrix.yaml", "r") as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

cameraMatrix = np.array(data['camera_matrix'])
disCoeff = np.array(data['dist_coeff'])

def mask_from_contours(ref_img, contours):
    mask = np.zeros(ref_img.shape, np.uint8)
    mask = cv2.drawContours(mask, contours, -1, (255,255,255), thickness=cv2.FILLED)
    return cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(output_filename,cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))
    return writer

def rot_params_rv(rvecs):
    R = cv2.Rodrigues(rvecs)[0]
    roll = 180*atan2(-R[2][1], R[2][2])/pi
    pitch = 180*asin(R[2][0])/pi
    yaw = 180*atan2(-R[1][0], R[0][0])/pi 
    rot_params= [yaw,pitch,roll]
    return rot_params

def reject_outliers(data, m = 2.):
    if len(data) > 1:
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        print(s)
        return [data[i] for i in range(len(s)) if s[i] < m]
    else:
        return data

def init_heading(frame, coords, prev_heading):
    lower_yellow = np.array([15, 30, 50])
    upper_yellow = np.array([35, 255, 255])
    xmin, ymin, xmax, ymax = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

    roi = frame[ymin:ymax, xmin:xmax]
    bb_area = (ymax-ymin)*(xmax-xmin)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thes = 0
    ct = 0

    for cnt in range(len(contours_yellow)):
        contour_area = cv2.contourArea(contours_yellow[cnt])
        if contour_area > bb_area/25:
            ellipse = cv2.fitEllipse(contours_yellow[cnt])
            (x,y),(MA,ma),angle = ellipse
            cv2.ellipse(roi,ellipse, (0,0,255), 2)
            rmajor = max(MA,ma)/2
            if angle > 90:
                ang = angle - 90
            else:
                ang = angle + 90
            print(angle)
            xc, yc = ellipse[0]
            x1 = xc + np.cos(np.deg2rad(ang))*rmajor
            y1 = yc + np.sin(np.deg2rad(ang))*rmajor
            x2 = xc + np.cos(np.deg2rad(ang+180))*rmajor
            y2 = yc + np.sin(np.deg2rad(ang+180))*rmajor
            cv2.line(roi, (int(x1),int(y1)), (int(x2),int(y2)), (0, 0, 255), 3)
            if abs(angle - prev_heading) >= abs(angle + 180 - prev_heading):
                angle += 180
            print("ang: " + str(angle))
            thes += angle
            ct += 1
    
    centroid = ((xmax+xmin)//2, (ymax+ymin)//2)

    ex = 50 * np.cos(np.deg2rad(thes+270)) + centroid[0]
    ey = 50 * np.sin(np.deg2rad(thes+270)) + centroid[1]
    cv2.line(frame, centroid, (int(ex), int(ey)), (0, 0, 255), 2)

    thes /= ct if ct != 0 else 1
    print(thes)
    
    return (thes+360)%360, frame


class App:
    def __init__(self, alg, mode, filename):
        self.referencePoints = []
        self.cropping = False
        self.roi = None # the marker object
        self.alg = alg # the algorithm to choose: sift/orb
        self.mode = mode # the way of choosing marker: static/screen capture
        self.filename = filename

    def click_and_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.referencePoints = [(x, y)]
            self.cropping = True
     
        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            self.referencePoints.append((x, y))
            self.cropping = False

    def draw(self, img, corners, imgpts, angle, coords):  #simple pose axes 
        centroid = coords[:2]
        center = np.array([(coords[2]+coords[0])//2,(coords[3]+coords[1])//2])
        corners = [corn + centroid for corn in corners] #np.array([img.shape[1]//2, img.shape[0]//2])
        corner = tuple(corners[0].ravel().astype(int)[:2]) #+ np.array([img.shape[1]//2, img.shape[0]//2]))
        img = cv2.polylines(img,corners,True,(255,255,255),2, cv2.LINE_AA)
        qx0 = center[0] + np.cos(np.deg2rad(angle - 90))*100 #- np.sin(np.deg2rad(angle))*5
        qy0 = center[1] + np.sin(np.deg2rad(angle - 90))*100 #angle #corner[1] + np.sin(np.deg2rad(angle))*(tuple(imgpts[0].ravel().astype(int))[0] - corner[0]) + np.cos(np.deg2rad(angle))*(tuple(imgpts[0].ravel().astype(int))[1] - corner[1])
        img = cv2.line(img, center, tuple([int(qx0), int(qy0)]), (255,0,0), 5) #blue
        #img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int) + center), (0,255,0), 5) #green
        #img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int) + center), (0,0,255), 5) #red
        return img

    def main(self):
        dirname = '../../template_data/'
        cap = cv2.VideoCapture(os.path.join(dirname, 'input_videos', self.filename)) #use 0 for webcam
        writer = create_video_writer(cap, os.path.join(dirname, 'output_videos', 'out_' + self.filename))
        counter = 0
        startFrame = 0
        isEnough = True
        running = []
        
        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        model = YOLO("../model_weights/segment/m_best.pt")
        heading = 0 #178.538983356
        prev_heading = 0
        prev_euler = 0
        rollover_euler = 0

        old_book = None
        old_mask = None
        old_coords = []
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            
            if counter >= startFrame:
                results = model.predict(source=frame.copy(), save_crop=True, save=True, save_txt=False, stream=True, conf=0.5, verbose=True, max_det = 1)

                for result in results:
                    if result.masks != None:
                        image = frame.copy()
                        mask_raw = result.masks[0].cpu().data.numpy().transpose(1, 2, 0) # only selecting biggest heron (fine for now)
                        mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))
                        h2, w2, c2 = result.orig_img.shape
                        mask = cv2.resize(mask_3channel, (w2, h2))
                        mask = cv2.inRange(mask, np.array([0,0,0]), np.array([0,0,1]))
                        mask = cv2.bitwise_not(mask)

                        thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
                        coords = np.column_stack(np.where(thresh.transpose() > 0))
                        thresh = cv2.erode(thresh, None, iterations=2)
                        thresh = cv2.dilate(thresh, None, iterations=2)
                        blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
                        cnts = cv2.findContours(blur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cnts = imutils.grab_contours(cnts)

                        #if len(cnts) > 0:
                        cnt_m = max(cnts, key=cv2.contourArea)
                        cnt = cv2.convexHull(cnt_m)
                        cv2.drawContours(image, [cnt], 0, (0,0,255), 2)
                        hull = mask_from_contours(frame, [cnt])
                        outputh = cv2.bitwise_and(result.orig_img, result.orig_img, mask=hull)

                        hull[hull > 0] = cv2.GC_PR_FGD
                        hull[hull == 0] = cv2.GC_BGD
                        fgModel = np.zeros((1, 65), dtype="float")
                        bgModel = np.zeros((1, 65), dtype="float")
                        (hull, bgModel, fgModel) = cv2.grabCut(frame, hull, None, bgModel, fgModel, iterCount=8, mode=cv2.GC_INIT_WITH_MASK)
                        outputMask = np.where((hull == cv2.GC_BGD) | (hull == cv2.GC_PR_BGD), 0, 1)
                        outputMask = (outputMask * 255).astype("uint8")
                        outputm = cv2.bitwise_and(result.orig_img, result.orig_img, mask=outputMask)

                        thresh = cv2.threshold(outputMask, 0, 255, cv2.THRESH_BINARY)[1]
                        M = cv2.moments(thresh)
                        cX = int(M["m10"] / (M["m00"] if M["m00"] != 0 else 1))
                        cY = int(M["m01"] / (M["m00"] if M["m00"] != 0 else 1))
                        (y, x) = np.where(outputMask == 255)
                        (topy, topx) = (np.min(y), np.min(x))
                        (bottomy, bottomx) = (np.max(y), np.max(x))
                        center = (cX, cY)
                        book = outputm[topy:bottomy+1, topx:bottomx+1]
                        book_mask = outputMask[topy:bottomy+1, topx:bottomx+1]
                        coords = (topx, topy, bottomx, bottomy)
                    else:
                        book = old_book
                        book_mask = old_mask
                        coords = old_coords

                if counter == startFrame:
                    if book is not None:
                        self.roi = ROI(book, self.alg)
                        matcher = Matcher(self.roi, self.alg, disCoeff, cameraMatrix)
                        heading, frame = init_heading(frame, coords, prev_heading)
                    else:
                        continue

                currentFrame = outputh.copy()
                cv2.namedWindow('webcam')
                matcher.setFrame(book)
                
                result = matcher.getCorrespondence()
                if result:
                    (src, dst, corners, M) = result
                else:
                    if book is not None:
                        print('Not enough points')
                        self.roi = ROI(book, self.alg)
                        matcher = Matcher(self.roi, self.alg, disCoeff, cameraMatrix)
                        heading, frame = init_heading(frame, coords, prev_heading)
                        running = [heading]
                        print("new template: " + str(heading))
                        rollover_euler = prev_euler
                        isEnough = False
                        cv2.putText(frame, "template switch", (300, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
                        cv2.imshow('webcam',imutils.resize(frame, width=350))
                        cv2.waitKey(1)
                        writer.write(frame)
                        continue
                    else:
                        continue

                (retvalCorner, rvecCorner, tvecCorner) = matcher.computePose(self.roi.getPoints3d(), corners)
                if counter == startFrame:
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

                running.append(heading)

                if len(running) > 10:
                    thes_new = np.mean(reject_outliers(running[-10:]))
                else:
                    thes_new = np.mean(reject_outliers(running))


                if retvalCorner:
                    axis = np.float32([[0.5,0,0], [0,0.5,0], [0,0,0.5]]).reshape(-1,3)
                    imgpts, jac = cv2.projectPoints(axis, rvecCorner, tvecCorner, cameraMatrix, disCoeff)

                    # re-draw the frame
                    currentFrame = self.draw(frame, [np.int32(corners)], imgpts, thes_new, np.array(coords))#np.array([coords[0],coords[1]])) 
                    fps = "Heading: " + str(int(thes_new + 360) %360)
                    cv2.putText(frame, fps, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
                    currentFrame = imutils.resize(currentFrame, width=350)
                    writer.write(frame)
                    # cv2.imshow('webcam', currentFrame)
                    # cv2.waitKey(1)
                else:
                    # cv2.waitKey(1)
                    continue

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    writer.release()
                    cv2.destroyAllWindows()
                    break

                old_book = book
                old_mask = book_mask
                old_coords = coords
            counter += 1

            if counter > startFrame:
                prev_heading = thes_new

        cap.release()
        writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = App("beblid_sift", "static", "IMG_8592.mov") 
    app.main()
