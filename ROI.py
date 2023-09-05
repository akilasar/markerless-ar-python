import cv2
import numpy as np
from matplotlib import pyplot as plt
from cnn_matching.lib.cnn_feature import cnn_feature_extract
from DALF_CVPR_2023.modules.models.DALF import DALF_extractor as DALF
import torch

class ROI(object):
	"""docstring for Marker"""
	def __init__(self, image, alg):
		self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		if alg == 'orb':
			# Initiate ORB detector
			orb = cv2.ORB_create()
			# find the keypoints and descriptors with ORB
			self.keypoints, self.descriptors = orb.detectAndCompute(self.image, None)
		elif alg == 'sift':
			# Initiate SIFT detector
			sift = cv2.SIFT_create()
			# find the keypoints and descriptors with SIFT
			self.keypoints, self.descriptors = sift.detectAndCompute(self.image, None)
		elif alg == 'beblid_sift':
			detector = cv2.SIFT_create(sigma=1)
			kpts1 = detector.detect(self.image, None)
			descriptor = cv2.xfeatures2d.BEBLID_create(6.25)
			self.keypoints, self.descriptors = descriptor.compute(self.image, kpts1)
		elif alg == 'teblid_sift':
			detector = cv2.SIFT_create()
			kpts1 = detector.detect(self.image, None)
			descriptor = cv2.xfeatures2d.TEBLID_create(6.25)
			self.keypoints, self.descriptors = descriptor.compute(self.image, kpts1)
		elif alg == 'beblid_orb':
			detector = cv2.ORB_create()
			kpts1 = detector.detect(self.image, None)
			descriptor = cv2.xfeatures2d.BEBLID_create(0.75)
			self.keypoints, self.descriptors = descriptor.compute(self.image, kpts1)
		elif alg == 'dalf':
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			dalf = DALF(dev = device)
			self.keypoints, self.descriptors = dalf.detectAndCompute(self.image)
		elif alg == "cnn":
			self.keypoints, _ , self.descriptors = cnn_feature_extract(self.image,  nfeatures = -1)
			
		width, height = self.image.shape

		# normalize
		maxSize = max(width, height)
		w = width/maxSize
		h = height/maxSize

		self.points2d = np.array([[0,0],[width,0],[width,height],[0,height]]) # corner points in 2d
		self.points3d = np.array([[0,0,0],[w,0,0],[w,h,0],[0,h,0]]) # corner points in 3d

	def getPoints2d(self):
		return self.points2d

	def getPoints3d(self):
		return self.points3d