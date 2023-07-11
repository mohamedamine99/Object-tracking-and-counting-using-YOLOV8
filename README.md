# Object-tracking-and-counting-using-YOLOV8

<p align="center">
    <img src="https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/blob/main/GIFs/Object%20Counting/yolov8l%20--%20traffic%202.gif" width=470
</p>

This repository contains the code for object detection, tracking, and counting using the YOLOv8 algorithm by ultralytics for object detection and the SORT (Simple Online and Realtime Tracking) algorithm for object tracking. The project provides code for both procedural and object-oriented programming implementations in Python.  

The OOP implementation is designed to be easily maintainable and customizable so that it can be further used for custom object detection, tracking, and counting.

For more details check the ultralytics YOLOv8 Github [repository](https://github.com/ultralytics/ultralytics) and the YOLOv8 python [documentation](https://docs.ultralytics.com/usage/python/#train).

### Features:

* **Object detection**: The YOLOv8 algorithm has been used to detect objects in images and videos. The algorithm is known for its fast and accurate performance.
* **Object tracking**: The SORT algorithm has been used for tracking the detected objects in real-time. SORT is a simple algorithm that performs well in real-time tracking scenarios.
* **Object counting**: The project also includes a module for counting the number of objects detected in a given image or video.
* **OOP approach**: The project has been implemented using object-oriented programming principles, making it modular and easy to understand.


## Navigating this repository

### Code files

* [YOLOv8_Object_Detection_procedural.ipynb](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/blob/main/YOLOv8_Object_Detection_procedural.ipynb) : The notebook provides code for object detection using YOLOv8, including different variants with different architectures and trade-offs between speed and accuracy. The code follows a procedural approach rather than object-oriented programming to make it simpler and easier to understand for beginners.

* [YOLOv8_Object_Detection_OOP.ipynb](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/blob/main/YOLOv8_Object_Detection_OOP.ipynb) : This notebook provides code for object detection using YOLOv8, including different variants with different architectures and trade-offs between speed and accuracy. The code follows an object-oriented approach rather than procedural programming to make it easier to understand, modify and maintain.

* [YOLOv8_Object_Counter_OOP.ipynb](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/blob/main/YOLOv8_Object_Counter_OOP.ipynb) : This notebook provides code for object detection, tracking and counting also using different YOLOv8 variants and an object-oriented approach.

* [YOLOv8_Object_Counter_OOP_v2.ipynb](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/blob/main/YOLOv8_Object_Counter_OOP_v2.ipynb) :This notebook provides code for object detection, tracking and counting also using different YOLOv8 variants and an object-oriented approach but the difference from [YOLOv8_Object_Counter_OOP.ipynb](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/blob/main/YOLOv8_Object_Counter_OOP.ipynb) is that the classes are imported as an external script named [yolo_detect_and_count.py](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/blob/main/yolo_detect_and_count.py) in order to avoid defining the classes inside the notebook and make it less cluttered with lines of code.

* [yolo_detect_and_count.py](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/blob/main/yolo_detect_and_count.py) : a python script that contains well-documented definitions for the `YOLOv8_ObjectDetector` and `YOLOv8_ObjectCounter` classes used respectively for detecting and counting objects.


* [sort.py](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/blob/main/sort.py) : a python module for object tracking using the SORT algorithm (SORT : Simple Online and Realtime Tracking) 
  * This script is a slightly modified version from the [original sort module](https://github.com/abewley/sort) by [abewley](https://github.com/abewley) , with some imports removed in order to fix some compatibility issues regarding pytorch and matplotlib backends.


### Results directories

* [results - Object Counting](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/tree/main/results%20-%20Object%20Counting) : This directory contains the video results for object counting in the following format : [yolo variant]--[original video name].avi

* [GIFs](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/tree/main/GIFs) : contains results for object detection and object counting on videos in GIF format.

* [results - custom display](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/tree/main/results%20-%20custom%20display) : This directory contains both video and image results for object detection using the custom display. The reason for this custom display is that it provides a simple alternative to the YOLOv8 native display which can sometimes be cumbersome. 

* [results - default display](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/tree/main/results%20-%20custom%20display) : This directory contains both video and image results for object detection using YOLOv8 default display. 

* [test vids](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/tree/main/test%20vids) : a directory that contains all testing videos.
* [test imgs](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/tree/main/test%20imgs) : a directory that contains all testing images.

### Other files

* [requirements.txt](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/blob/main/requirements.txt) : Contains the modules required by the sort module
* [coco.names](https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8/blob/main/coco.names) : Contains the labels of the 80 default classes for the COCO dataset.
