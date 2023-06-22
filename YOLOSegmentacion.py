import cv2
import pandas as pd
from ultralytics import YOLO


model=YOLO('yolov8s-seg.pt')
#model=YOLO('yolov8m-pose.pt')

cv2.namedWindow('YOLO Segmentacion', cv2.WINDOW_NORMAL)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)

cap=cv2.VideoCapture("peoplecount2.mp4")


while True:

    ret,frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1800, 1020))
    results = model.predict(source=frame, show=True, conf=0.5)
    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
    #    print(px)
    list = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        #print(c)
        list.append([x1, y1, x2, y2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, c, (x1, y1-50), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 2)

    # bbox_idx = tracker.update(list)
    # for bbox in bbox_idx:
    #     x3, y3, x4, y4, id = bbox
    #     cx = int(x3 + x4) // 2
    #     cy = int(y3 + y4) // 2
    #
    #
    #     cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
    #     cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
    #     cv2.putText(frame, str(int(id)), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)



    cv2.imshow("YOLO Segmentacion", frame)


    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()