from ultralytics import YOLO
import cv2

model = YOLO('../yolo_weights/yolov8n.pt')
results = model("baza_img/Getty_school_bus_safety_LARGE_Agnieszka-Kirinicjanow-56a13ea95f9b58b7d0bd5ff0.jpg", show = True)
cv2.waitKey(0)