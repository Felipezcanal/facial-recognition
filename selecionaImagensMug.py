import os
#for imghdr.what to find wheter a file is image
#['pbm','pgm','ppm','png','jpeg','tiff','bmp','webp']
import imghdr
import cv2
import sys
import shutil

import math


def deal_with_data_ck(list):
	if (len(list) > 0):



		threshold = math.floor((len(list) - 1) * 0.4)  # how many pictures be considered neutral in directory
		threshold2 = math.floor((len(list) - 1) * 0.6)
		for x in list:
			if imghdr.what(x) in ['png', 'jpeg', 'jpg']:  # Makes sure file is .png image
				y = x[4:8]
				if ( (int(y) < threshold) or (int(y) > threshold2)):
					os.remove(x)


def trav_dir(dirpath):
	os.chdir(dirpath)
	dir_list = os.listdir()

	# travers current directory and if directoy found call itself
	for x in dir_list:
		if(x == "neutral"):
			continue
		if (os.path.isdir(x)):
			trav_dir(x)
		# imghdr.what return mime type of the image
		elif (imghdr.what(x) in ['jpg', 'jpeg', 'png']):
			deal_with_data_ck(dir_list)
			break

	# reached directory with no directory
	os.chdir('./..')

# trav_dir('/media/felipe/02F2FC09F2FBFF2B/IMAGENSSSSSS/mug')
trav_dir('/app/mug')

