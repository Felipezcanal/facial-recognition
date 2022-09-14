import shutil
import csv
from pprint import pprint
import imghdr
import cv2
import os
import os.path

face_cascade = cv2.CascadeClassifier('haar.xml')

folder = "/media/felipe/UbuntuHD/dataset/images/1/Manually_Annotated_Images/"

to = "/media/felipe/UbuntuHD/dataset/images/dest/"


# shutil.move("este-arquivo", "/tmp")


with open('affect.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:


        imgpath = folder+row[1]
        if not os.path.isfile(imgpath):
            continue

        gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(gray, 1.25, 5)
        if len(faces) == 0:
            print(imgpath)  # Change parameters of detectMultiScale or manually crop the image
            os.remove(imgpath)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (500, 500), interpolation=cv2.INTER_AREA)
            cv2.imwrite(imgpath, roi_gray)


            split = row[1].split("/")
            shutil.move(imgpath, to+row[0]+'/'+split[1])
            # pprint(row)
            pprint(split)