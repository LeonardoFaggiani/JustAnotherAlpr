# Just Another ALPR

This is a proof of concept of how to train my own `YOLO` model to detect vehicle license plates.

`Darknet` is used as a framework and the `YOLO` algorithm to detect license plates, `YOLOv4-tiny` is used because the main idea is to be able to detect license plate in real time.

On the other hand, I used `Python` to draw and manipulate the steaming of the video.


Installation & Requirements
------------

1) Install the Darknet framework, I recommend the following repository
https://github.com/hank-ai/darknet
2) Clone https://github.com/LeonardoFaggiani/JustAnotherAlpr.git
3) Go to the darknet folder and modify the darknet.py file on line 328 "{path/to/your}/darknet.dll" replace {path/to/your} with the path where the darknet installer is located, save it.
4) Go to the nn/license-plate path and modify the license-plate.data file and you must tell darknet where your license-plate.names file is, for example if you cloned the JustAnotherAlpr repository on the C drive, it would be C://JustAnotherAlpr/nn/license-plate/license-plate.names

Usage
-----

## Video/Webcam/Images

### Process a Video File
    python darknet_video.py --input Testing-Driving.mp4 --weights ..\nn\license-plate\license-plate.weights --config_file ..\nn\license-plate\license-plate.cfg --data_file ..\nn\license-plate\license-plate.data

### Use Display for Real-Time Detection
    python darknet_video.py --input 0 --weights ..\nn\license-plate\license-plate.weights --config_file ..\nn\license-plate\license-plate.cfg --data_file ..\nn\license-plate\license-plate.data

### Use Webcam for Real-Time Detection
    python darknet_video.py --input 1 --weights ..\nn\license-plate\license-plate.weights --config_file ..\nn\license-plate\license-plate.cfg --data_file ..\nn\license-plate\license-plate.data

### Save Processed Video
    python darknet_video.py --input Testing-Driving.mp4 --out_filename processed_Testing-Driving.avi --weights ..\nn\license-plate\license-plate.weights --config_file ..\nn\license-plate\license-plate.cfg --data_file ..\nn\license-plate\license-plate.data


## TODO

* [x] Upload differents kind of images to train the model.
* [x] Create Model
* [x] Draw boxes in video streaming
* [ ] OCR 🛠️ (In Progress)
* [ ] Information of the owner of the vehicle license (Simulation)