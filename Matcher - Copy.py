'''
Author: Zhaorui Chen 2017

Code based on OpenCV doc - python tutorial.
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from cnn_matching.lib.cnn_feature import cnn_feature_extract
from DALF_CVPR_2023.modules.models.DALF import DALF_extractor as DALF
from DALF_CVPR_2023.modules.tps import RANSAC

import torch

MIN_MATCH_COUNT = 10

class Matcher:
	def __init__(self, roi, alg, distCoeffs, cameraMatrix):
		self.roi = roi # a ROI pattern object
		self.alg = alg
		self.cameraMatrix = cameraMatrix
		self.distCoeffs = distCoeffs

	def setFrame(self, frame):
		self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	def getCorrespondence(self):
		# Get the previously extracted keypoints and descriptors for the marker object
		kp1 = self.roi.keypoints
		des1 = self.roi.descriptors

		if self.alg == 'orb':
			orb = cv2.ORB_create()
			kp2, des2 = orb.detectAndCompute(self.frame, None) # kp2: keypoints from the frame/captured image
			FLANN_INDEX_LSH = 6
			index_params= dict(algorithm = FLANN_INDEX_LSH,
				   table_number = 6, # 12
				   key_size = 12,     # 20
				   multi_probe_level = 1) #2
		elif self.alg == 'sift':
			FLANN_INDEX_KDTREE = 1
			sift = cv2.SIFT_create()
			kp2, des2 = sift.detectAndCompute(self.frame, None) # kp2: keypoints from the frame/captured image
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		elif self.alg == 'beblid_sift':
			detector = cv2.SIFT_create()
			kpts1 = detector.detect(self.frame, None)
			descriptor = cv2.xfeatures2d.BEBLID_create(6.25)
			kp2, des2 = descriptor.compute(self.frame, kpts1)
		elif self.alg == 'teblid_sift':
			detector = cv2.SIFT_create()
			kpts1 = detector.detect(self.frame, None)
			descriptor = cv2.xfeatures2d.TEBLID_create(6.25)
			kp2, des2 = descriptor.compute(self.frame, kpts1)
		elif self.alg == 'beblid_orb':
			detector = cv2.ORB_create()
			kpts1 = detector.detect(self.frame, None)
			descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
			kp2, des2 = descriptor.compute(self.frame, kpts1)
		elif self.alg == "cnn":
			kp2, _ , des2 = cnn_feature_extract(self.frame,  nfeatures = -1)
			FLANN_INDEX_KDTREE = 1
			index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
			search_params = dict(checks=40)
		elif self.alg == 'dalf':
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			dalf = DALF(dev = device)
			kp2, des2 = dalf.detectAndCompute(self.frame)

		search_params = dict(checks = 50)

		if self.alg == 'dalf':
			matcher = cv2.BFMatcher(crossCheck = True)
			matches = matcher.match(des1, des2)
			src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
			tgt_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
			inliers = RANSAC.nr_RANSAC(src_pts, tgt_pts, device, thr = 0.2)
			good = [matches[i] for i in range(len(matches)) if inliers[i]]
		else:
			if self.alg[1:6] != "eblid":
				flann = cv2.FlannBasedMatcher(index_params, search_params)
			else:
				flann = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

			# simple nn, can be refined with cross validation
			# des1: descriptors from the marker
			# des2: descriptors from the frame
			if(des1 is not None and len(des1)>2 and des2 is not None and len(des2)>2):
				matches = flann.knnMatch(des1,des2,k=2) # [ ..., [<DMatch 0x104a34c30>, <DMatch 0x104a34c50>], ... ]

				good = [] # good is a container for 3D points found in scene
				for m_n in matches:
					if len(m_n)!=2:
						continue
					(m, n) = m_n
					if m.distance < 0.7*n.distance:
						good.append(m) # m is from the scene/captured image <DMatch 0x104a34c30>

				if len(good)>MIN_MATCH_COUNT:
					if self.alg == "cnn":
						src_pts = np.float32([ cv2.KeyPoint(kp1[m.queryIdx][0], kp1[m.queryIdx][1], 1).pt for m in good ]).reshape(-1,1,2) # n*(1*2) 
						dst_pts = np.float32([ cv2.KeyPoint(kp2[m.trainIdx][0],  kp2[m.trainIdx][1],  1).pt for m in good ]).reshape(-1,1,2) # n*(1*2)
					else:
						src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) # n*(1*2) 
						dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2) # n*(1*2)

				M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) # M is the transformation matrix, thresholding at 5
				matchesMask = mask.ravel().tolist()

				h,w = self.roi.image.shape
				pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2) # 4 corners of the roi pattern image

				try:
					corners = cv2.perspectiveTransform(pts,M) # points2d for the frame image
				except:
					print('No matching points after homography estimation')
					return
				
				return (src_pts, dst_pts, corners, M)

			else:
				print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
				matchesMask = None
				return None

	def computePose(self, src, dst):
		''' find the camera pose for the current frame, by solving PnP problem '''

		'''
		cv2.solvePnP(objectPoints, imagePoints, 
			cameraMatrix, distCoeffs[, rvec[, tvec[, useExtrinsicGuess[, flags]]]]) 
		→ retval, rvec, tvec
		'''
		retval, rvec, tvec = cv2.solvePnP(src, dst, self.cameraMatrix, self.distCoeffs)
		return (retval, rvec, tvec)

