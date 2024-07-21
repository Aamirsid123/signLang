import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import os
import cv2
import numpy as np
#import math

# for any laptop
# model = tf.keras.models.load_model(os.path.join('model', 'model.keras'))

# if running on m1

model = tf.keras.models.load_model(os.path.join('model', 'model.keras'), compile=False)
model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

class_mapping = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


def predict_image_class(img):
    try:
        if img is None:
            print(f"Failed to read image")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img)
        predicted_label = np.argmax(predictions)
        return predicted_label
    except Exception as e:
        print(f"Error predicting image : {e}")
        return None


result = []
# Iterate over images and predict
cap = cv2.VideoCapture(0)

imgSize = 300
offset = 20

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            break

        print(class_mapping[predict_image_class(frame)])

        cv2.imshow("Image", frame)
        cv2.waitKey(1)
    except Exception as e:
        print("Error loading image:", e)

# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# import time
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import os
#
# cap = cv2.VideoCapture(0)
#
# detector = HandDetector()
# classifier = Classifier("model/keras_model.h5", "model/labels.txt")
# #classifier = tf.keras.models.load_model(os.path.join('model', 'model.keras'), compile=False)
# #classifier.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
# imgSize = 300
# offset = 20
# counter = 0
# labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
#            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
# folder = "Photo/R"
# while True:
#     try:
#       success, img = cap.read()
#       hands, img = detector.findHands(img)
#       if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#
#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
#         imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
#         imgCropShape = imgCrop.shape
#         aspectRatio = h/w
#         if aspectRatio > 1:
#             k = imgSize/h
#             wCal = math.ceil(k*w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgSize-wCal)/2)
#             imgWhite[0:imgResizeShape[0], wGap:wCal+wGap] = imgResize
#             # prediction, index = classifier.getPrediction(imgWhite)
#             # print(prediction, index)
#
#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgSize - hCal) / 2)
#             imgWhite[hGap:hCal + hGap, 0:imgResizeShape[1]] = imgResize
#             # prediction, index = classifier.getPrediction(imgWhite)
#             # print(prediction, index)
#
#         cv2.imshow("ImageCrop", imgCrop)
#         cv2.imshow("ImageWhite", imgWhite)
#
#       cv2.imshow("Image", img)
#       # cv2.waitKey(1)
#     except Exception as e:
#          print(e)
#     key = cv2.waitKey(1)
#     if key == ord('s'):
#          counter += 1
#          cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
#          print(counter)


# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from cvzone.HandTrackingModule import HandDetector
# import os
# import cv2
# import numpy as np
# import math
# #for any laptop
# #model = tf.keras.models.load_model(os.path.join('model', 'model.keras'))
#
# #if running on mac
# model = tf.keras.models.load_model(os.path.join('model', 'model.keras'), compile=False)
# model.compile(optimizer='adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
#
#
#
#
# data = tf.keras.utils.image_dataset_from_directory(os.path.join('Photo','archive (4)','asl_alphabet_train','asl_alphabet_train'))
# class_mapping = data.class_names
# print(class_mapping)
#
# def predict_image_class(img):
#     try:
#         if img is None:
#             print(f"Failed to read image")
#             return None
#         print("error free")
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = cv2.resize(img, (64, 64))
#         img = np.expand_dims(img, axis=0)
#
#         predictions = model.predict(img)
#         predicted_label = np.argmax(predictions)
#         return predicted_label
#     except Exception as e:
#         print(f"Error predicting image : {e}")
#         return None
#
#
# result = []
# # Iterate over images and predict
# cap = cv2.VideoCapture(0)
# detector = HandDetector()
#
# imgSize = 300
# offset = 20
#
# while True:
#     try:
#       success, img = cap.read()
#       hands, img = detector.findHands(img)
#       if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#
#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
#         imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
#         imgCropShape = imgCrop.shape
#         aspectRatio = h/w
#         if aspectRatio > 1:
#             k = imgSize/h
#             wCal = math.ceil(k*w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgSize-wCal)/2)
#             imgWhite[0:imgResizeShape[0], wGap:wCal+wGap] = imgResize
#            # print(predict_image_class(imgWhite))
#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgSize - hCal) / 2)
#             imgWhite[hGap:hCal + hGap, 0:imgResizeShape[1]] = imgResize
#             #print(predict_image_class(imgWhite))
#
#         cv2.imshow("ImageCrop", imgCrop)
#         cv2.imshow("ImageWhite", imgWhite)
#
#       cv2.imshow("Image", img)
#       cv2.waitKey(1)
#     except Exception as e:
#          print("Error loading image:", e)
