import time

import numpy as np
from numpy import size

np.random.seed(1337)  # for reproducibility
import keras
from keras.models import model_from_json
import cv2
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
from model import model


predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
fa = FaceAligner(predictor, desiredFaceWidth=500)
face_cascade = cv2.CascadeClassifier('haar.xml')

image = "teste.jpg"

nb_classes = 7
# input image dimensions
img_rows, img_cols = 150, 150


def deal_with_image(img_path):
    # gray = cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.25, 5)
    if len(faces) == 0:
        return 0
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (500, 500), interpolation=cv2.INTER_AREA)
        # print(cv2.imwrite(imgpath , roi_gray ))
        return roi_gray


def align(img):
    # img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_cinza = img
    faces = detector(img_cinza)
    if len(faces) > 0:
        for rosto in faces:
            shape_68 = predictor(img, rosto)
            shape = face_utils.shape_to_np(shape_68)
            rosto_Alinhado = fa.align(img, img_cinza, rosto)
            return rosto_Alinhado
    return 0


def landmarks(img):
    dets = detector(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    if (len(dets) != 1):
        return 0
    for k, d in enumerate(dets):

        shape = predictor(img, d)

        yfactor = 12
        img2 = np.zeros((500, 500, 3))
        for i in range(0, 67):
            if i == 16 or i == 21 or i == 26 or i == 35 or i == 41 or i == 47 or i == 59 or i < 16:
                continue
            cv2.line(img2, (shape.parts()[i].x, shape.parts()[i].y - yfactor),
                     (shape.parts()[i + 1].x, shape.parts()[i + 1].y - yfactor), (0, 255, 0), 1)

        cv2.line(img2, (shape.parts()[35].x, shape.parts()[35].y - yfactor),
                 (shape.parts()[30].x, shape.parts()[30].y - yfactor), (0, 255, 0), 1)
        cv2.line(img2, (shape.parts()[41].x, shape.parts()[41].y - yfactor),
                 (shape.parts()[36].x, shape.parts()[36].y - yfactor), (0, 255, 0), 1)
        cv2.line(img2, (shape.parts()[47].x, shape.parts()[47].y - yfactor),
                 (shape.parts()[42].x, shape.parts()[42].y - yfactor), (0, 255, 0), 1)
        cv2.line(img2, (shape.parts()[59].x, shape.parts()[59].y - yfactor),
                 (shape.parts()[48].x, shape.parts()[48].y - yfactor), (0, 255, 0), 1)
        cv2.line(img2, (shape.parts()[67].x, shape.parts()[67].y - yfactor),
                 (shape.parts()[60].x, shape.parts()[60].y - yfactor), (0, 255, 0), 1)
        return img2

input_shape = (img_rows, img_cols, 1)
# size of pooling area for max pooling
pool_size = (4, 4)
# convolution kernel size
kernel_size = (6, 6)

modelo = model(input_shape)


# load weights into new model
# model.load_weights("todos.weights.best.hdf5")
modelo.load_weights("weights.best.hdf5")
print("Loaded model from disk")

# evaluate loaded model on test data
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# modelo.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


def handleVideo():
    # cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)
    # cv2.namedWindow('gray', cv2.WINDOW_KEEPRATIO)

    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            time.sleep(5)
            pass
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        # frame = cv2.imread("orig.jpg")

        cv2.imshow("original", frame)
        cv2.waitKey(5)

        img = deal_with_image(frame)
        if size(img) <= 1:
            continue
        # try:
        img = align(img)
        # except:
        #     continue

        if size(img) <= 1:
            continue
        img = landmarks(img)

        if (size(img) <= 1):
            continue

        cv2.imshow("landmarks", img)

        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
        # cv2.imshow("asd", img)
        # cv2.waitKey(0)
        img = img.flatten()  # changes 100x100 to 10000 in a row major fashion
        img = img / 25500  # normalization of data [for sigmoid neurons(0-1)]
        img = img.astype(np.float32)

        img = img.reshape(1, img_rows, img_cols, 1)
        score = modelo.predict([img])

        results.append(score)
        # print(score)

        if len(results) == 10:
            res = np.sum(results, axis=0)
            print(res)
            results.pop(0)


            index_max = np.argmax(res)

            cv2.imshow("emoji", emojis[index_max])
            emotion = ["Neutral", "Anger", "Contempt", "Disgust", "Fear", "Happy", "Sadness", "Surprise"]
            print(emotion[index_max])

        cv2.waitKey(100)

results = []


video_capture = cv2.VideoCapture(1)

cv2.namedWindow("original", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("landmarks", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("emoji", cv2.WINDOW_KEEPRATIO)

emojis = []
for i in range(8):
    emojis.append(cv2.imread("emojis/"+str(i)+".png"))

handleVideo()