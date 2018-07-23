
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import pickle

#載入臉部辨識
face_cascade = cv2.CascadeClassifier('E:/Python/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('E:/Python/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')

#載入訓練後YML
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

#開啟標籤、名稱對照
labels={"person_name" : 1}
with open("labels.pickle","rb") as f:
    og_labels = pickle.load(f)
    #變換位子
    labels = {v:k for k,v in og_labels.items()}

#擷取視訊
cap = cv2.VideoCapture(0)

while True:
    #獲取視訊及返回狀態
    ret , frame = cap.read()
    #影像灰階
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #檢測視訊中的人臉，並用vector儲存人臉的座標、大小（用矩形表示)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    #臉部檢測
    #x,y,w,h抓臉部的座標
    for (x,y,w,h) in faces: 
        #print(x,y,w,h)
        
        #臉部灰階區塊
        roi_gray = gray[y:y+h,x:x+w]
        #臉部彩色區塊
        roi_color = frame[y:y+h,x:x+w]
    
        #相似度達45-85之間  顯示ID
        id_, confidence  = recognizer.predict(roi_gray)
        if confidence   >= 45 :   #and confidence  <= 85:
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255 , 255)
            stroke = 2
            cv2.putText(frame, name , (x,y) , font , 1 , color , stroke , cv2.LINE_AA)
        
       
        #=========看灰階狀況，並儲存 Start    ==========#   
        
        #img_item = "my-image.png"
        #cv2.imwrite(img_item,roi_gray)
        
        #=========看灰階狀況，並儲存 End    ==========#   
    
        #框線顏色 BGR 0-255
        color = ( 0 , 0, 255)
        #框線寬度
        stroke = 2
        #框寬度
        end_cord_x = x + w
        #框長度
        end_cord_y = y + h
        #畫框線
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)
    
    
    #顯示視訊
    cv2.imshow('frame',frame)
    
    #刷新畫面  超過1秒跳出或點 'q' 跳出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
#關閉視訊
cap.release()
#關閉所有畫面
cv2.destroyAllWindows()

