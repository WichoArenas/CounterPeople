import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *


model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture("peoplecount2.mp4")
#cap=cv2.VideoCapture(0)


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0
tracker=Tracker()   


#area1 = [(150, 545),(150,600),(370,640),(355,572)]
area1 = [(227, 390),(238,408),(339,363),(329,349)]
area2 = [(403,327),(418,366),(565,331),(541,293)]


area1_c=set()
area2_c=set()

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue


    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.boxes
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        list.append([x1,y1,x2,y2])
            
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        results=cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
        if results >= 0 :
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
            cv2.circle(frame,(x4,y4),5,(255,0,255),-1)
            cv2.putText(frame,str(int(id)),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
            area1_c.add(id)
        results1=cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
        if results1 >= 0 :
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
            cv2.circle(frame,(x4,y4),5,(255,0,255),-1)
            cv2.putText(frame,str(int(id)),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
            area2_c.add(id)
    first=len(area1_c)
    second=len(area2_c)
    cv2.putText(frame,str(first),(50,80),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)
    cv2.putText(frame, str(second), (50, 180), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 0, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

