#================================================================
#  To learn how to Develop Advance YOLOv4 Apps - Then check out:
#  https://augmentedstartups.info/yolov4release
#================================================================ 
from ctypes import *
import os
import cv2
import darknet
import glob

def convertBack(x, y, w, h):								# Convert from center coordinates to bounding box coordinates
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
	#================================================================
    # 1. Purpose : Vehicle Counting
    #================================================================    
    if len(detections) > 0:    								# If there are any detections
        car_detection = 0
        for detection in detections:						# For each detection
            name_tag = detection[0].decode()				# Decode list of classes 
            if name_tag == 'car':							# Filter detections for car class
	            x, y, w, h = detection[2][0],\
	                detection[2][1],\
	                detection[2][2],\
	                detection[2][3]  						# Obtain the detection coordinates
	            xmin, ymin, xmax, ymax = convertBack(
	                float(x), float(y), float(w), float(h))  	# Convert to bounding box coordinates
	            pt1 = (xmin, ymin)								# Format Coordinates for Point 1 and 2
	            pt2 = (xmax, ymax)
	            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)  	# Draw our rectangles
            car_detection += 1 								# Increment to the next detected car
        cv2.putText(img,
                    "Total cars %s" % str(car_detection), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    [0, 255, 50], 2)						# Place text to display the car count
    return img 												# Return Image with detections
    #=================================================================#


netMain = None
metaMain = None
altNames = None


def YOLO(image_list):

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
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
    i = 0
    while True:
        image = cv2.imread(image_list[i])
        width = image.shape[1]
        height = image.shape[0]

        # Create an image we reuse for each detect
        darknet_image = darknet.make_image(width, height, 3)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb,
                                       (width, height),
                                       interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, image_rgb.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, image_rgb)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Output', image)
        cv2.waitKey(0)
        i += 1
    cv2.destroyAllWindows()

if __name__ == "__main__":
	#================================================================
    # 2. Purpose : Get the list of Input Image Files
    #================================================================  
    image_path = "D:\YOLOv4\darknet-master\Input\\"			#  Directory of the image folder
    image_list = glob.glob(image_path + "*.jpg")			#  Get list of Images
    print(image_list)		
    #=================================================================#									
    YOLO(image_list)
