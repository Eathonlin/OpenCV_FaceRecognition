
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np
from PIL import Image
import pickle

#目前檔案路徑
BASE_DIR = os.path.dirname(os.path.abspath("__file__"))
#路經+pics
image_dir = os.path.join(BASE_DIR,"pics")

face_cascade = cv2.CascadeClassifier('E:/Python/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}

y_labels = []
x_train = []

#尋找資料夾中所有目錄及檔名
for root, dirs, files in os.walk(image_dir):
    for file in files:
        # 副檔名為png&jpg
        if file.endswith("PNG") or file.endswith("png") or file.endswith("jpg") or file.endswith("JPG"):
            path = os.path.join(root, file)
            #label 顯示檔案所在的資料夾名稱
            label = os.path.basename(root).replace(" ","-").lower()
            #print(label,path)
            
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #print(label_ids)
            #y_labels.append(label)
            #x_train.append(path)
            
            #用PIL將圖像轉換成L模式 L = R * 299/1000 + G * 587/1000+ B * 114/1000
            pil_image = Image.open(path).convert("L")
            
            #固定照片大小
            size = (550 , 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            
            image_array = np.array(pil_image, "uint8")
            #print(image_array)
            
            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)
             
#寫入對照表
with open("labels.pickle","wb") as f:
    pickle.dump(label_ids,f)
          
#訓練好蟹入YML
recognizer.train(x_train,np.array(y_labels))
recognizer.write("trainner.yml")

