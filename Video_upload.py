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



df = pd.DataFrame(columns=['time', 'emotion'])

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


class videoclass():
    def __init__(self):
        super().__init__()
        path = easygui.fileopenbox(default='*')
        cap = cv2.VideoCapture(path)
        time_rec = datetime.now()
        #faces = face_cascade.detectMultiScale(gray, 1.1, 4)


        while True:
            time_rec = datetime.now()

            ret, frame = cap.read()
            if ret:
                
                gray = cv2.flip(frame, 1)

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

                    cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)
                    cv2.imshow('Video', gray)
                    cv2.resizeWindow('Video', 1000, 600)

                    if cv2.waitKey(1) == 27:  # press ESC to break
                        cap.release()
                        cv2.destroyAllWindows()
                        break

                    



if __name__=='__main__':
    app = QApplication(sys.argv)
    video_recog = videoclass()
    video_recog.show()
    app.exit(app.exec_())



"""print(df.head())
print(df.shape)

import matplotlib.pyplot as plt

emo_data = df.groupby('emotion').size()
print(emo_data, '\n')

emotion_dict_count = {'anger': 0, 'contempt': 0, 'disgust': 0,
                      'fear': 0, 'happiness': 0,
                      'sadness': 0, 'surprise': 0}

for i in df['emotion']:
    emotion_dict_count[str(i)] += 1

    emo_count = [x for x in emotion_dict_count.values()]
    emo_name = [x for x in emotion_dict_count.keys()]


for i,j in zip(emo_count, emo_name):
    if i == 0:
        emo_count.remove(i)
        emo_name.remove(j)



plt.pie(x=emo_count, labels=emo_name, autopct='%1.2f', startangle=90)

# print('\n emo_count :',emo_count)
# print('\n emo_name :',emo_name)


plt.title("Emotions Recorded ")

plt.show()"""
