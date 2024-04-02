import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../videos/cars.mp4")

model = YOLO("../yolo_weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("mask2.png")

# Tracker
tracker = Sort(max_age=20,min_hits=3,iou_threshold=0.3)

limit = [20,447,773,447]
totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion,stream=True)    # stream => using generators => more efficient
    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding boxs
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100

            # Class name
            cls = int(box.cls[0]) 
            currentClass = classNames[cls]
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and conf > 0.3:
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale = 0.6, thickness = 1, offset = 3)
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9 , rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections,currentArray))

    trackerResults = tracker.update(detections)

    cv2.line(img,(limit[0],limit[1]),(limit[2],limit[3]),(0,0,255),5)

    for result in trackerResults:
        x1,y1,x1,y2,id =  result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        # cvzone.cornerRect(img, (x1, y1, w, h), l=8 ,rt=5 , colorR=(255,0,0))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(0, y1)), scale=1, thickness=3,
                           offset=10)
        cx, cy = x1+w//4 , y1+h//4
        cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED)

        if limit[0] < cx < limit[2] and limit[1]-30 < cy < limit[3]+30:
            if totalCount.count(id) == 0:
                totalCount.append(id)

    cvzone.putTextRect(img, f'Count{len(totalCount)}', (50,50))
    cv2.imshow("Image", img)
    # cv2.imshow("Image Region", imgRegion)

    cv2.waitKeyEx(0)
