# -*- coding: utf-8 -*-
import cv2
import numpy as np
import dlib
import pickle
import os
import imghdr
from imutils import face_utils
from imutils.face_utils import FaceAligner
from random import shuffle, randint

shape_predictor_68 = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
fa = FaceAligner(shape_predictor_68, desiredFaceWidth=500)
	
def trav_dir(dirpath):
	os.chdir(dirpath)
	dir_list = os.listdir()
	
	#travers current directory and if directoy found call itself
	for x in dir_list:
		if(os.path.isdir(x)):
			print(x)
			trav_dir(x)
		#imghdr.what return mime type of the image
		elif(imghdr.what(x) in ['jpeg', 'png', 'jpg']):
			img = cv2.imread(x)			
			img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			faces = detector(img_cinza)
			if len(faces) > 0:
				for rosto in faces:
					shape_68 = shape_predictor_68(img, rosto)
					shape = face_utils.shape_to_np(shape_68)
					rosto_Alinhado = fa.align(img, img_cinza, rosto)
					cv2.imwrite(x, rosto_Alinhado)
			else:
				print(x)  # Change parameters of detectMultiScale or manually crop the image
				os.remove(x)

	#reached directory with no directory
	os.chdir('./..')

# to = "/media/felipe/UbuntuHD/dataset/images/dest/"
trav_dir('/app/mug')
# trav_dir('dataset/ck')
