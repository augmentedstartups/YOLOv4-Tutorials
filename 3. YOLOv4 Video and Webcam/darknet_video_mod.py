from ctypes import *                                               # Import libraries
import math
import random
import os
import cv2
import numpy as np
import time
import darknet


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    # Colored labels dictionary
    color_dict = {
        'person' : [0, 255, 255], 'bicycle': [238, 123, 158], 'car' : [24, 245, 217], 'motorbike' : [224, 119, 227],
        'aeroplane' : [154, 52, 104], 'bus' : [179, 50, 247], 'train' : [180, 164, 5], 'truck' : [82, 42, 106],
        'boat' : [201, 25, 52], 'traffic light' : [62, 17, 209], 'fire hydrant' : [60, 68, 169], 'stop sign' : [199, 113, 167],
        'parking meter' : [19, 71, 68], 'bench' : [161, 83, 182], 'bird' : [75, 6, 145], 'cat' : [100, 64, 151],
        'dog' : [156, 116, 171], 'horse' : [88, 9, 123], 'sheep' : [181, 86, 222], 'cow' : [116, 238, 87],'elephant' : [74, 90, 143],
        'bear' : [249, 157, 47], 'zebra' : [26, 101, 131], 'giraffe' : [195, 130, 181], 'backpack' : [242, 52, 233],
        'umbrella' : [131, 11, 189], 'handbag' : [221, 229, 176], 'tie' : [193, 56, 44], 'suitcase' : [139, 53, 137],
        'frisbee' : [102, 208, 40], 'skis' : [61, 50, 7], 'snowboard' : [65, 82, 186], 'sports ball' : [65, 82, 186],
        'kite' : [153, 254, 81],'baseball bat' : [233, 80, 195],'baseball glove' : [165, 179, 213],'skateboard' : [57, 65, 211],
        'surfboard' : [98, 255, 164],'tennis racket' : [205, 219, 146],'bottle' : [140, 138, 172],'wine glass' : [23, 53, 119],
        'cup' : [102, 215, 88],'fork' : [198, 204, 245],'knife' : [183, 132, 233],'spoon' : [14, 87, 125],
        'bowl' : [221, 43, 104],'banana' : [181, 215, 6],'apple' : [16, 139, 183],'sandwich' : [150, 136, 166],'orange' : [219, 144, 1],
        'broccoli' : [123, 226, 195],'carrot' : [230, 45, 209],'hot dog' : [252, 215, 56],'pizza' : [234, 170, 131],
        'donut' : [36, 208, 234],'cake' : [19, 24, 2],'chair' : [115, 184, 234],'sofa' : [125, 238, 12],
        'pottedplant' : [57, 226, 76],'bed' : [77, 31, 134],'diningtable' : [208, 202, 204],'toilet' : [208, 202, 204],
        'tvmonitor' : [208, 202, 204],'laptop' : [159, 149, 163],'mouse' : [148, 148, 87],'remote' : [171, 107, 183],
        'keyboard' : [33, 154, 135],'cell phone' : [206, 209, 108],'microwave' : [206, 209, 108],'oven' : [97, 246, 15],
        'toaster' : [147, 140, 184],'sink' : [157, 58, 24],'refrigerator' : [117, 145, 137],'book' : [155, 129, 244],
        'clock' : [53, 61, 6],'vase' : [145, 75, 152],'scissors' : [8, 140, 38],'teddy bear' : [37, 61, 220],
        'hair drier' : [129, 12, 229],'toothbrush' : [11, 126, 158]
    }
    
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        name_tag = str(detection[0].decode())
        for name_key, color_val in color_dict.items():
            if name_key == name_tag:
                color = color_val 
                xmin, ymin, xmax, ymax = convertBack(
                float(x), float(y), float(w), float(h))
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                cv2.rectangle(img, pt1, pt2, color, 1)
                cv2.putText(img,
                            detection[0].decode() +
                            " [" + str(round(detection[1] * 100, 2)) + "]",
                            (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 2)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():
   
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4.cfg"                                 # Path to cfg
    weightPath = "./yolov4.weights"                                 # Path to weights
    metaPath = "./cfg/coco.data"                                    # Path to meta data
    if not os.path.exists(configPath):                              # Checks whether file exists otherwise return ValueError
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:                                             # Checks the metaMain, NetMain and altNames. Loads it in script
        netMain = darknet.load_net_custom(configPath.encode( 
            "ascii"), weightPath.encode("ascii"), 0, 1)             # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)                                      # Uncomment to use Webcam
    cap = cv2.VideoCapture("test2.mp4")                             # Local Stored video detection - Set input video
    frame_width = int(cap.get(3))                                   # Returns the width and height of capture video
    frame_height = int(cap.get(4))
    # Set out for video writer
    out = cv2.VideoWriter(                                          # Set the Output path for video writer
        "./Demo/output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (frame_width, frame_height))

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(frame_width, frame_height, 3) # Create image according darknet for compatibility of network
    while True:                                                      # Load the input frame and write output frame.
        prev_time = time.time()
        ret, frame_read = cap.read()                                 # Capture frame and return true if frame present
        # For Assertion Failed Error in OpenCV
        if not ret:                                                  # Check if frame present otherwise he break the while loop
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)      # Convert frame into RGB from BGR and resize accordingly
        frame_resized = cv2.resize(frame_rgb,
                                   (frame_width, frame_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())                # Copy that frame bytes to darknet_image

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)    # Detection occurs at this line and return detections, for customize we can change the threshold.                                                                                   
        image = cvDrawBoxes(detections, frame_resized)               # Call the function cvDrawBoxes() for colored bounding box per class
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)                                    # Display Image window
        cv2.waitKey(3)
        out.write(image)                                             # Write that frame into output video
    cap.release()                                                    # For releasing cap and out. 
    out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":  
    YOLO()                                                           # Calls the main function YOLO()