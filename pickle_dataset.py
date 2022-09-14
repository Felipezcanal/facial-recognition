import imghdr
import math
import os
import pickle
import random

import cv2
import numpy as np


class PickleData(object):

    def __init__(self):
        # using list because append is costly on numpy array/ need to hardcode the array size
        self.neutral_img = []
        self.anger_img = []
        self.contempt_img = []
        self.disgust_img = []
        self.fear_img = []
        self.happy_img = []
        self.sadness_img = []
        self.surprise_img = []
        self.count = 0

    '''
    traverse directory tree if a directory found with
    image then find if .txt file exist if exist and 
    start collecting data
    '''

    def deal_with_data_ck(self, list):
        if (len(list) > 0):

            if any(".txt" in s for s in list):  # check whether directory have emotion label

                threshold = math.floor((len(list) - 1) * 0.3)  # how many pictures be considered neutral in directory
                # threshold = 3

                qtd = len(list)
                ignore_rate = 0.2
                ignore = math.floor(qtd * ignore_rate)
                for x in list:  # find emotion label and read it
                    if ".txt" in x:
                        text_file = open(x, 'rb')
                        text = int(float(text_file.readline()))
                        text_file.close()
                        break

                for x in list:
                    if imghdr.what(x) in ['png', 'jpeg', 'jpg']:  # Makes sure file is .png image
                        img = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, image_size_wanted, interpolation=cv2.INTER_AREA)
                        img = img.flatten()  # changes 100x100 to 10000 in a row major fashion
                        img = img / 25500  # normalization of data [for sigmoid neurons(0-1)]
                        img = img.astype(np.float32)
                        # if int(x[10:-4]) <= threshold: #Image name are of type 'S005_001_00000002.png' looks at the part '00000002'
                        if (x[-12:-4] == 'mirrored'):
                            y = x[10:-17]
                        else:
                            y = x[10:-4]
                        if int(y) <= (math.floor(qtd * 0.3) - math.floor(
                                ignore / 2)):  # Image name are of type 'S005_001_00000002.png' looks at the part '00000002'
                            self.count += 1
                            self.neutral_img.append(img)
                        elif (int(y) >= (math.ceil(qtd * 0.3) + math.ceil(ignore / 2))):
                            # else:
                            self.count += 1
                            if (text == 1):
                                self.anger_img.append(img)
                            # elif (text == 2):
                            #     self.contempt_img.append(img)
                            elif (text == 3):
                                self.disgust_img.append(img)
                            elif (text == 4):
                                self.fear_img.append(img)
                            elif (text == 5):
                                self.happy_img.append(img)
                            elif (text == 6):
                                self.sadness_img.append(img)
                            elif (text == 7):
                                self.surprise_img.append(img)

                        # print(self.count)

    '''
    same code as in haar_apply.py but only break statement
    after the execution of elif becuase once we reache 
    the direcory with images we will process all the 
    image in directoy with function deal_with_data_ck()
    '''

    def deal_with_data(self, list):
        if (len(list) > 0):

            if any(".txt" in s for s in list):  # check whether directory have emotion label

                for x in list:  # find emotion label and read it
                    if ".txt" in x:
                        text_file = open(x, 'rb')
                        # print(float(text_file.readline()), os.path.abspath(__file__))
                        text = int(float(text_file.readline()))
                        text_file.close()
                        break

                for x in list:
                    if imghdr.what(x) in ['jpg', 'jpeg', 'png']:  # Makes sure file is .jpg or .jpeg image
                        img = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, image_size_wanted, interpolation=cv2.INTER_AREA)
                        img = img.flatten()  # changes 100x100 to 10000 in a row major fashion
                        img = img / 25500  # normalization of data [for sigmoid neurons(0-1)]
                        img = img.astype(np.float32)

                        self.count += 1
                        if (text == 0):
                            self.neutral_img.append(img)
                        elif (text == 1):
                            self.anger_img.append(img)
                        # elif (text == 2):
                        #     self.contempt_img.append(img)
                        elif (text == 3):
                            self.disgust_img.append(img)
                        elif (text == 4):
                            self.fear_img.append(img)
                        elif (text == 5):
                            self.happy_img.append(img)
                        elif (text == 6):
                            self.sadness_img.append(img)
                        elif (text == 7):
                            self.surprise_img.append(img)
                    # print(self.count)

    '''
    same code as in haar_apply.py but only break statement
    after the execution of elif becuase once we reache 
    the direcory with images we will process all the 
    image in directoy with function deal_with_data_ck()
    '''

    def trav_dir(self, dirpath, ck):
        os.chdir(dirpath)
        dir_list = os.listdir()

        # travers current directory and if directoy found call itself
        for x in dir_list:
            if (os.path.isdir(x)):
                self.trav_dir(x, ck)
            # imghdr.what return mime type of the image
            elif (imghdr.what(x) in ['jpg', 'jpeg', 'png']):
                if (ck == 1):
                    self.deal_with_data_ck(dir_list)
                else:
                    self.deal_with_data(dir_list)
                break

        # reached directory with no directory
        os.chdir('./..')

    def pack_data(self):

        # biglist = [self.neutral_img, self.anger_img, self.contempt_img, self.disgust_img, self.fear_img, self.happy_img,
        #            self.sadness_img, self.surprise_img]

        random.shuffle(self.neutral_img)
        random.shuffle(self.anger_img)
        random.shuffle(self.disgust_img)
        random.shuffle(self.fear_img)
        random.shuffle(self.happy_img)
        random.shuffle(self.sadness_img)
        random.shuffle(self.surprise_img)

        biglist = [self.neutral_img, self.anger_img, self.disgust_img, self.fear_img, self.happy_img, self.sadness_img, self.surprise_img]


        self.training_data = []
        self.validation_data = []
        self.test_data = []

        self.training_txt = []
        self.validation_txt = []
        self.test_txt = []

        for x, y in zip(biglist, range(0, 7)):
            length = len(x)
            print('Lenght:::::::::', length)

            self.training_data.append(x[0:math.ceil(length * 0.7)])  # 80 percent Image of each emotion for training
            self.training_txt.append(y * np.ones(shape=(len(x[0:math.ceil(length * 0.7)]), 1),
                                                 dtype=np.int8))  # Generating Corresponding Label

            self.validation_data.append(x[math.ceil(length * 0.7):math.floor(length * 0.85)])  # 10 percent
            self.validation_txt.append(
                y * np.ones(shape=(len(x[math.ceil(length * 0.7):math.floor(length * 0.85)]), 1), dtype=np.int8))

            self.test_data.append(x[math.floor(length * 0.85):length])  # 10 percent
            self.test_txt.append(y * np.ones(shape=(len(x[math.floor(length * 0.85):length]), 1), dtype=np.int8))

        del biglist

        self.training_data = np.vstack(
            self.training_data)  # np.vstack(list_of_array) converts the list_of_numpy_arrays into a single numpy array
        self.validation_data = np.vstack(self.validation_data)
        self.test_data = np.vstack(self.test_data)
        # print(self.training_txt)
        self.training_txt = np.vstack(self.training_txt)
        self.validation_txt = np.vstack(self.validation_txt)
        self.test_txt = np.vstack(self.test_txt)

    def pickle_data(self):
        pickle.dump(((self.training_data, self.training_txt), (self.validation_data, self.validation_txt),
                     (self.test_data, self.test_txt)), open('/home/felipe/Documents/rec-facial/dataset/mug-150-70-30.p', 'wb'))

    def start_pickling(self):

        # self.trav_dir('dataset/nosso_alinhado', 0)
        # self.trav_dir('/app/dataset/ck-redimensionado', 1)  # name of the directory

        self.trav_dir('./dataset/mug-pronto-com-mirror', 0)  # name of the directory
        self.pack_data()
        self.pickle_data()


# import p

image_size_wanted = (150, 150)
count = 0
p1 = PickleData()
p1.start_pickling()
print(p1.count)
print("neutro " + str(len(p1.neutral_img)))
print("anger " + str(len(p1.anger_img)))
print("happy " + str(len(p1.happy_img)))
print("contempt " + str(len(p1.contempt_img)))
print("disgust " + str(len(p1.disgust_img)))
print("fear " + str(len(p1.fear_img)))
print("sadness " + str(len(p1.sadness_img)))
print("surprise " + str(len(p1.surprise_img)))




# 25256
# neutro 2569
# anger 4323
# happy 4001
# contempt 0
# disgust 3624
# fear 3214
# sadness 3465
# surprise 4060
