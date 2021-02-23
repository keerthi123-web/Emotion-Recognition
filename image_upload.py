import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PyQt5.QtWidgets import QFileDialog
from datetime import datetime
import seaborn as sns
import easygui


#df = pd.DataFrame(columns=['time', 'emotion'])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('model.h5')
emotion_dict = {0: 'anger', 1: 'contempt', 2: 'disgust',
                3: 'fear', 4: 'happiness',
                5: 'sadness', 6: 'surprise'}



def convert_image(image):
    image_arr = []
    pic = cv2.resize(image, (48, 48))
    image_arr.append(pic)
    image_arr = np.array(image_arr)
    image_arr = image_arr.astype('float32')
    image_arr /= 255
    ans = model.predict_classes(image_arr)[0]
    return ans

class mainclass():
    def __init__(self):
        super().__init__()
        path = easygui.fileopenbox(default='*')
        #path = QFileDialog.getOpenFileName(self, 'open file',' ',"Image files (*.jpg *.gif)")
        gray = cv2.imread(path)
        time_rec = datetime.now()
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            prediction = int(convert_image(roi_gray))
            #print(prediction)
            emotion = emotion_dict[prediction]

            df = pd.DataFrame(columns=['time', 'emotion'])
            df = df.append({'time': time_rec, 'emotion': emotion}, ignore_index=True)
            #print("hi")

            cv2.putText(gray, emotion, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2 , cv2.LINE_AA
                        )

            cv2.namedWindow('Image', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Image', gray)
            cv2.resizeWindow('Image', 1000, 600)

            if cv2.waitKey(1) == 27:  # press ESC to break
                cap.release()
                cv2.destroyAllWindows
                break



#class_instance = mainclass()
#class_instance.image_upload()
if __name__=='__main__':
    app = QApplication(sys.argv)
    img_recog = mainclass()
    img_recog.show()
    app.exit(app.exec_())
