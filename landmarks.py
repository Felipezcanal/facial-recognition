
import sys
import os
import os.path
import dlib
import glob
import cv2
from inspect import getmembers
from pprint import pprint
import numpy as np
import imghdr

def landmarks(f):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    # win.clear_overlay()
    # win.set_image(img)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    if(len(dets) != 1):
        os.remove(f)
        return
    for k, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        # print("Part 0: {}, Part 1: {} , Part 2: {} , Part 3: {} , Part 4: {} , Part 5: {} ...".format(shape.part(0),
        #                                           shape.part(1),
        #                                           shape.part(2),
        #                                           shape.part(3),
        #                                           shape.part(4),
        #                                           shape.part(5)))

        # Draw the face landmarks on the screen.
        # win.add_overlay(shape)
        # cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 255), 2)
        # cv2.line(img, shape.part(0), shape.part(1), (255, 0, 0), 5)
        yfactor = 12
        img2 = np.zeros((500,500,3))
        for i in range(0, 67):
            if(i == 16 or i == 21 or i == 26 or i == 35 or i == 41 or i == 47 or i == 59 or i < 16): # < 16 tira a parte do contorno do rosto  (deixa so olhos, nariz e boca
                continue
            cv2.line(img2, (shape.parts()[i].x, shape.parts()[i].y - yfactor), (shape.parts()[i+1].x, shape.parts()[i+1].y - yfactor), (0, 255, 0), 1)

        cv2.line(img2, (shape.parts()[35].x, shape.parts()[35].y - yfactor), (shape.parts()[30].x, shape.parts()[30].y - yfactor), (0, 255, 0), 1)
        cv2.line(img2, (shape.parts()[41].x, shape.parts()[41].y - yfactor), (shape.parts()[36].x, shape.parts()[36].y - yfactor), (0, 255, 0), 1)
        cv2.line(img2, (shape.parts()[47].x, shape.parts()[47].y - yfactor), (shape.parts()[42].x, shape.parts()[42].y - yfactor), (0, 255, 0), 1)
        cv2.line(img2, (shape.parts()[59].x, shape.parts()[59].y - yfactor), (shape.parts()[48].x, shape.parts()[48].y - yfactor), (0, 255, 0), 1)
        cv2.line(img2, (shape.parts()[67].x, shape.parts()[67].y - yfactor), (shape.parts()[60].x, shape.parts()[60].y - yfactor), (0, 255, 0), 1)
        cv2.imwrite(f, img2)

        print('landmarks' + f)

# if len(sys.argv) != 3:
#     print(
#         "Give the path to the trained shape predictor model as the first "
#         "argument and then the directory containing the facial images.\n"
#         "For example, if you are in the python_examples folder then "
#         "execute this program by running:\n"
#         "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
#         "You can download a trained facial shape predictor from:\n"
#         "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
#     exit()


# predictor_path = sys.argv[1]
# faces_folder_path = sys.argv

predictor_path = "shape_predictor_68_face_landmarks.dat"
# faces_folder_path = "/media/felipe/UbuntuHD/dataset/images/teste"
# faces_folder_path = "/media/felipe/02F2FC09F2FBFF2B/IMAGENSSSSSS/mug"
faces_folder_path = "/app/mug"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# win = dlib.image_window()

def trav_dir(dirpath):
    print(dirpath)
    os.chdir(dirpath)
    dir_list = os.listdir()
    dir_list.sort(reverse=True)

	#travers current directory and if directoy found call itself
    for x in dir_list:
        if(os.path.isdir(x)):
            trav_dir(x)
		#imghdr.what return mime type of the image
        elif(imghdr.what(x) in ['jpeg', 'jpg', 'png']):

            if (x == "asd.jpg"):
                os.remove("asd.jpg")
            else:
                landmarks(x)

	#reached directory with no directory
    os.chdir('./..')

trav_dir(faces_folder_path)


