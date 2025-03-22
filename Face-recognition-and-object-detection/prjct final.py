# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 19:30:01 2024

@author: ACER
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 31 20:06:52 2024

@author: ACER
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 31 01:05:26 2024

@author: ACER
"""

import face_recognition
import cv2
import os
import glob
import numpy as np
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import serial.tools.list_ports
import time


window1 = Tk()
window1.geometry('750x720')
window1.title('chemseddine IMAGE TO TEXT')
window1.resizable(True, True)
window1.configure(bg='white')

frame = Frame(window1, bg='white', width=750, height=720)


from gtts import gTTS
import pygame

import tempfile
import shutil

def text_to_speech(text, file_name):
    """
    تحويل النص إلى كلام وحفظه كملف صوتي، ثم تشغيله وإعادة تسميته بعد التشغيل.
    
    :param text: النص المراد تحويله إلى كلام
    :param file_name: اسم الملف لحفظ الصوت (بدون الامتداد)
    """
    # تحديد اللغة
    language = 'ar'
    
    # تحويل النص إلى كلام باستخدام gTTS
    tts = gTTS(text=text, lang=language, slow=False)
    
    # إنشاء ملف مؤقت لحفظ الصوت
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
        temp_file_name = temp_file.name
        tts.save(temp_file_name)
    
    # تهيئة pygame
    pygame.mixer.init()
    pygame.init()
    
    # تحميل الصوت من الملف المؤقت
    pygame.mixer.music.load(temp_file_name)
    
    # تشغيل الصوت
    pygame.mixer.music.play()
  
    
    # تأخير قصير للتأكد من أن الصوت قد انتهى بالفعل
    time.sleep(0.3)    
    # تحديد اسم الملف الجديد
    new_file_name = f"{file_name}.mp3"
    
    # إعادة تسمية الملف المؤقت
    try:
        shutil.move(temp_file_name, new_file_name)
    except OSError as e:
        print(f"Error: {e}")
        print("Failed to rename the file.")
    


# استخدام الدالة



# حذف الملف المؤقت بعد تشغيله



####### From Image #######
def ImgFile():
   img = cv2.imread('person.png')

   classNames = []
   classFile = 'coco.names'

   with open(classFile, 'rt') as f:
      classNames = f.read().rstrip('\n').split('\n')

   configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
   weightpath = 'frozen_inference_graph.pb'

   net = cv2.dnn_DetectionModel(weightpath, configPath)
   net.setInputSize(320 , 230)
   net.setInputScale(1.0 / 127.5)
   net.setInputMean((127.5, 127.5, 127.5))
   net.setInputSwapRB(True)

   classIds, confs, bbox = net.detect(img, confThreshold=0.5)
   print(classIds, bbox)

   for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
      cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
      cv2.putText(img, classNames[classId-1], (box[0] + 10, box[1] + 20), 
                  cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)


   cv2.imshow('Output', img)
   cv2.waitKey(0)
######################################

####### From Video or Camera #######
def Camera():
   cam = cv2.VideoCapture(0)

   cam.set(3, 740)
   cam.set(4, 580)

   classNames = []
   classFile = 'coco.names'

   with open(classFile, 'rt') as f:
      classNames = f.read().rstrip('\n').split('\n')

   configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
   weightpath = 'frozen_inference_graph.pb'

   net = cv2.dnn_DetectionModel(weightpath, configPath)
   net.setInputSize(320 , 230)
   net.setInputScale(1.0 / 127.5)
   net.setInputMean((127.5, 127.5, 127.5))
   net.setInputSwapRB(True)

   while True:
      success, img = cam.read()
      classIds, confs, bbox = net.detect(img, confThreshold=0.5)
      print(classIds, bbox)

      if len(classIds) !=0:
         for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[classId-1], (box[0] + 10, box[1] + 20), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
            text_to_speech(classNames[classId-1], classNames[classId-1])
            
            


      cv2.imshow('Output', img)
      
      key = cv2.waitKey(2)
      if key == 27:
            break
   cam.release()
   cv2.destroyAllWindows()
######################################


## Call ImgFile() Function for Image Or Camera() Function for Video and Camera
# ImgFile()


known_face_encodings = []
known_face_names = []

        # Resize frame for a faster speed
frame_resizing = 0.15


# Load the model
model_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} not found. Make sure you have downloaded it and placed it in the correct directory.")

known_face_encodings = []
known_face_names = []

# Resize frame for a faster speed
frame_resizing = 0.15

def f2():
    def load_encoding_images(images_path):
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store file name and file encoding
            known_face_encodings.append(img_encoding)
            known_face_names.append(filename)
        print("Encoding images loaded")
        print("m=", known_face_names)

    def detect_known_faces(frame):
        small_frame = cv2.resize(frame, (0, 0), fx=frame_resizing, fy=frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                value = 42  # القيمة التي تريد إرسالها إلى Arduino في حالة التعرف
            else:
                value = 40  # القيمة التي تريد إرسالها إلى Arduino في حالة عدم التعرف
            face_names.append(name)

        face_locations = np.array(face_locations)
        face_locations = face_locations / frame_resizing
        return face_locations.astype(int), face_names

    s = load_encoding_images("pr l3")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        face_locations, face_names = detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
            text_to_speech(name, name)
           
        frame = cv2.resize(frame, (700, 400))
        cv2.imshow("frame", frame)
        key = cv2.waitKey(2)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def th(pho, a, b):
    image = Image.open(pho)
    photo_width = a
    photo_height = b
    image = image.resize((photo_width, photo_height), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    return photo

ex = "youcef brik.jpeg"
img = th(ex, 750, 720)

bg_label = Label(frame, image=img)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

button = Button(frame, text="START", font=('Franklin Gothic Medium', 12), borderwidth=0, highlightthickness=0, bd=5, relief='solid', command=f2)
button.config(width=15, height=2)
button.place(relx=0.5, rely=0.5, anchor=CENTER)
button = Button(frame, text="Detection", font=('Franklin Gothic Medium', 12), borderwidth=0, highlightthickness=0, bd=5, relief='solid', command=Camera)
button.config(width=15, height=2)
button.place(relx=0.5, rely=0.8, anchor=CENTER)

frame.pack(fill='both', expand=True)

window1.mainloop()
