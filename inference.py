from ultralytics import YOLO
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import frame
import numpy as np

model = YOLO("model/best_n.pt")

def infer_img(img):
    #SORT Tracking object recieved from frame object
    mot_tracker = frame.tracker.get_tracker()

    #infer on the frame
    results = model.predict(source=img, conf = frame.tracker.yolo_confidence/100, half = True) # save predictions as labels
    boxes = np.array(results[0].boxes.data.cpu())
    track_bb_ids = mot_tracker.update(boxes[:,:5])
    for j in range(len(track_bb_ids.tolist())):
                coords = track_bb_ids.tolist()[j]
                x1,y1,x2,y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(img, (x1,y1), (x2,y2),(0,0,255),1)
                cv2.circle(img, (center_x,center_y), 5, (0, 0, 255), -1)

    return img