import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image
from datetime import datetime
import seaborn as sns
import easygui
from image_upload import mainclass
from Video_upload import videoclass

df = pd.DataFrame(columns=['time', 'emotion'])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = load_model('model.h5')
emotion_dict = {0: 'anger', 1: 'contempt', 2: 'disgust',
                3: 'fear', 4: 'happiness',
                5: 'sadness', 6: 'surprise'}


#load model
model = model_from_json(open("fer.json", "r").read())

model.load_weights('fer.h5')

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap=cv2.VideoCapture(0)



def convert_image(image):
    image_arr = []
    pic = cv2.resize(image, (48, 48))
    image_arr.append(pic)
    image_arr = np.array(image_arr)
    image_arr = image_arr.astype('float32')
    image_arr /= 255
    ans = model.predict_classes(image_arr)[0]
    return ans

class LoadingScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(1900, 800)
        #self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.CustomizeWindowHint)
        
        #self.label_animation = QLabel(self)
        #self.label_animation.setAlignment(Qt.AlignCenter)

        label = QLabel()
	
        vbox = QVBoxLayout()
        vbox.addWidget(label)
        vbox.setAlignment(Qt.AlignCenter)
        self.setLayout(vbox)

        label.setStyleSheet("padding-top :-10px;"
                            "padding-right :500px;"
                            "padding-bottom :150px;")
        label.move(900, 600) 
        
        self.movie = QMovie('oie_316937GzLwlO7w.gif')
        label.setMovie(self.movie)

        timer = QTimer(self)
        
        self.startAnimation()
        timer.singleShot(3000, self.stopAnimation)

        self.show()

    def startAnimation(self):
        self.movie.start()

    def stopAnimation(self):
        self.movie.stop()
        self.close()
        

class AppDemo(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Emotion Facial Recognition")
        self.setFixedSize(1370, 800)

        oImage = QImage("facial-recognition-img2.jpg")
        sImage = oImage.scaled(QSize(1370,800))
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(sImage))
        self.setPalette(palette)

        
        label_1 = QLabel('<font size=10 color=Black>Emotion Recognition</font>', self)
        label_1.setStyleSheet("border : 4px solid black;")
        width = 279
        label_1.setFixedWidth(width)
        height = 60
        label_1.setFixedHeight(height)
        label_1.resize(190, 100)
        label_1.setFont(QFont('Arial',weight=QFont.Bold))

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        label_1.setGraphicsEffect(shadow)
       
        label_1.setGeometry(30,330,250,50)

        label_2 = QLabel('<font size=6 color=Black>Choose Option for Emotion Recognition</font>', self)
        width = 500
        label_2.setFixedWidth(width)
        height = 60
        label_2.setFixedHeight(height)
        label_2.resize(190, 100)
        label_2.setFont(QFont('Arial',weight=QFont.Bold))
        label_2.move(950, 200)
        
        pybutton = QPushButton('Choose Webcam', self)
        pybutton.setStyleSheet("background-color :  #99bede;"
                               "color : white;"
                               "border : none;")
        pybutton.setFont(QFont('Arial',12,weight=QFont.Bold))
        pybutton.clicked.connect(self.CaptureVideoMethod)
        pybutton.resize(200,32)
        pybutton.move(1052, 280)

        pybutton_1 = QPushButton('Choose Video File', self)
        pybutton_1.setStyleSheet("background-color :  #99bede;"
                                 "color : white;"
                                 "border : none;")
        pybutton_1.setFont(QFont('Arial',12,weight=QFont.Bold))
        pybutton_1.clicked.connect(self.new_videoUpload_emotionMethod)
        pybutton_1.resize(200,32)
        pybutton_1.move(1052, 360)

        pybutton_2 = QPushButton('Choose Image File', self)
        pybutton_2.setStyleSheet("background-color :  #99bede;"
                                 "color : white;"
                                 "border : none;")
        pybutton_2.setFont(QFont('Arial',12,weight=QFont.Bold))
        pybutton_2.clicked.connect(self.new_imageUpload_emotionMethod)
        pybutton_2.resize(200,32)
        pybutton_2.move(1052, 435)

        self.loading_screen = LoadingScreen()

        self.show()

    def CaptureVideoMethod(self):
        while True:
            ret,test_img=cap.read()# captures frame and returns boolean value and captured image
            if not ret:
                continue
            gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


            for (x,y,w,h) in faces_detected:
                cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
                roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
                roi_gray=cv2.resize(roi_gray,(48,48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)

                #find max indexed array
                max_index = np.argmax(predictions[0])

                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                predicted_emotion = emotions[max_index]

                cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            resized_img = cv2.resize(test_img, (1000, 700))
            cv2.imshow('Facial emotion analysis ',resized_img)



            if cv2.waitKey(1) == 27:#wait until 'q' key is pressed
                cap.release()
                cv2.destroyAllWindows
                break

        
    
    def new_videoUpload_emotionMethod(self):
        self.new_video_recog_window = videoclass()
        self.new_video_recog_window.show()
        
    def new_imageUpload_emotionMethod(self):
        self.new_img_recog_window = mainclass()
        self.new_img_recog_window.show()
    """def ImageFileMethod(self):
        path = easygui.fileopenbox(default='*')
        gray = cv2.imread(path)
        time_rec = datetime.now()
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            print(roi_gray)
            prediction = int(convert_image(roi_gray))
            print(prediction)
            emotion = emotion_dict[prediction]

            df = df.append({'time': time_rec, 'emotion': emotion}, ignore_index=True)
            print("hi")

            cv2.putText(gray, emotion, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2 , cv2.LINE_AA
                        )

            cv2.namedWindow('Video', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Video', gray)
            cv2.resizeWindow('Video', 1000, 600)

            if cv2.waitKey(1) == 27:  # press ESC to break
                break

            else:
                break"""
        

        
        
if __name__=='__main__':
    app = QApplication(sys.argv)
    demo = AppDemo()
    demo.show()
    app.exit(app.exec_())


    
