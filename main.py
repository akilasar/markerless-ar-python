import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from ROI import ROI
from Matcher import Matcher
import argparse
import imutils
import yaml
import os

with open("../utils/calibration_boards/calibration_matrix.yaml", "r") as stream:
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
	 
	# def draw(self, img, corners, imgpts): #3D Cube superimposition
	# 	imgpts = np.int32(imgpts).reshape(-1,2)
	# 	# draw ground floor in green
	# 	img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
	# 	# draw pillars in blue color
	# 	for i,j in zip(range(4),range(4,8)):
	# 		img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
	# 		# draw top layer in red color
	# 		img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
	# 	return img

	def draw(self, img, corners, imgpts):  #simple pose axes 
		corner = tuple(corners[0].ravel().astype(int)[:2])
		img = cv2.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 5)
		img = cv2.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 5)
		img = cv2.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 5)
		img = cv2.polylines(img,corners,True,255,3, cv2.LINE_AA)
		return img

	def main(self):
		dirname = '../../template_data/'
		cap = cv2.VideoCapture(os.path.join(dirname, 'input_videos', self.filename)) #use 0 for webcam
		writer = create_video_writer(cap, os.path.join(dirname, 'output_videos', 'out_' + self.filename))

		if self.mode == 'capture':
			while True:
	 
				# Capture frame-by-frame
				ret, frame = cap.read()

				currentFrame = frame.copy()
				cv2.namedWindow("choose marker")
				cv2.setMouseCallback("choose marker", self.click_and_crop)
				
				# Display the resulting frame
				cv2.imshow('choose marker',frame)

				# if there are two reference points, then crop the region of interest
				# from teh image and display it
				if len(self.referencePoints) == 2:
					cropImage = currentFrame[self.referencePoints[0][1]:self.referencePoints[1][1], \
							self.referencePoints[0][0]:self.referencePoints[1][0]]
					cv2.rectangle(currentFrame, self.referencePoints[0], self.referencePoints[1], (0, 255, 255), 2)
					# initialize a marker object for the marker
					self.roi = ROI(cropImage, self.alg)

					cv2.imshow('choose marker',currentFrame)
					cv2.waitKey(1000)
					cv2.destroyWindow('choose marker')
					break

				if cv2.waitKey(1) & 0xFF == ord('q'):
					cap.release()
					cv2.destroyAllWindows()
					break
		else:
			roi = cv2.imread('image0.jpg')
			self.roi = ROI(roi, self.alg)

		cv2.waitKey(100)
		matcher = Matcher(self.roi, self.alg, disCoeff, cameraMatrix)

		while True:
			ret, frame = cap.read()

			if not ret:
				break

			currentFrame = frame.copy()
			mirrorFrame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)
	
			cv2.namedWindow('webcam')
			#currentFrame = imutils.resize(currentFrame, width=500)
			#cv2.imshow('webcam', currentFrame)
			matcher.setFrame(currentFrame)
			
			result = matcher.getCorrespondence()
			if result:
				(src, dst, corners) = result
			else:
				print('Not enough points')
				cv2.waitKey(1)
				continue

			(retvalCorner, rvecCorner, tvecCorner) = matcher.computePose(self.roi.getPoints3d(), corners)
			if retvalCorner:
				print(rvecCorner)
				print(tvecCorner)
				#axis = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0], [0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])
				axis = np.float32([[0.5,0,0], [0,0.5,0], [0,0,0.5]]).reshape(-1,3)
				imgpts, jac = cv2.projectPoints(axis, rvecCorner, tvecCorner, cameraMatrix, disCoeff)

				# re-draw the frame
				currentFrame = cv2.polylines(currentFrame,[np.int32(corners)],True,255,3, cv2.LINE_AA)
				currentFrame = self.draw(currentFrame, [np.int32(corners)], imgpts)

				#currentFrame = imutils.resize(currentFrame, width=500)
				writer.write(currentFrame)
				cv2.imshow('webcam', currentFrame)
				cv2.waitKey(1)
				
			else:
				cv2.waitKey(1)
				continue

			if cv2.waitKey(1) & 0xFF == ord('q'):
				cap.release()
				cv2.destroyAllWindows()
				break

		cap.release()
		writer.release()
		cv2.destroyAllWindows()


if __name__ == '__main__':

	parser = argparse.ArgumentParser()	
	parser.add_argument("mode", help="choose the program mode from {static, capture}, static will use the image named roi.png in the same directory")
	parser.add_argument("algorithm", help="choose the feature extraction algorithm from {sift, orb}")
	parser.add_argument("filename", help="add filename of video input")
	args = parser.parse_args()
	
	app = App(args.algorithm, args.mode, args.filename)
	app.main()