# 1. YOLOv4 Prerequisites Installation Documentation

## Step 1
-	Cloned repository from official Github of Alexyab [link]
## Step 2
-	Copied OpenCV ffmpeg and world dlls for 4.1.0 version(can be used for OpenCV video processing) near to darknet.exe
## Step 3
-	Copied cuDNN dll(for speedup neural network process) near to darknet.exe  
## Step 4
-	Change the cuda version if you’re not using Cuda 10.0.
-	I’m using 10.2 so made change in two files
1)	darknet.vcxproj [Line:55]
2)	yolo_cpp_dll.vcxproj [Line:309]
-	Make it save
## Step 5
Compile yolo_cpp_dll.vcxproj with updated Cuda version
-	Open yolo_cpp_dll.vcxproj with VS
-	Open VS and Select Release and x64
-	Right click on yolo_cpp_dll and select build
-	Should be 1 succeeded without fail
	[Optional]
-	If not succeeded make sure you followed upper steps properly
 

## Step 6 Compile darknet.sln
-	Open darknet.sln with VS
-	Set Release and x64
-	Right click on darknet and select properties
-	Make Four changes
-	C/C++ -> General -> Additional Include Directories:
C:\OpenCV\build\install\include Click Ok and Apply
If you followed my instructions then path should be same
	      -	C/C++ -> Preprocessor -> Preprocessor Definitions
		Remove CUDNN_HALF as Yolo v4/3 uses Tensor cores
We don’t need to use it for 1080Ti
[Optional]
If you have GPU with Tensor Cores (nVidia Titan V / Tesla V100 / DGX-2 and later) then add CUDNN_HALF for speedup in detection 3x and training 2x.
-	CUDA C/C++ -> Device -> Code Generation 
Remove compute_75, sm_75 if you’re not using Cuda 10.0
                 - 	Linker -> General
		Add lib from OpenCV build
		C:\OpenCV\build\install\x64\vc16\lib
-	Right click on darknet and select build
-	After the successfully build try to import darknet in Python from x64 directory
Should be imported otherwise right click on darknet and select clean the build and try to follow upper steps properly.

## Step 7 Run Detection on Images

`darknet.exe detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights`
It’ll ask for the input path of the image

## Step 8 Run Detection on Videos

`darknet.exe detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights test.mp4`


