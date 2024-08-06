# Just Another ALPR

This is a proof of concept of how to train my own `YOLO` model to detect vehicle license plates.

`Darknet` is used as a framework and the `YOLO` algorithm to detect license plates, `YOLOv4-tiny` is used because the main idea is to be able to detect license plate in real time.

On the other hand, I used `Python` to draw and manipulate the steaming of the video.

<<<<<<< HEAD
![Alt Text](./darknet/samples/LicensePlate.gif)
=======
[![Alt Text](./darknet/samples/LicensePlate.gif)]
>>>>>>> 6bcb30b207a47b4e5eb3d8ee5860e3736855263c


Installation & Requirements
------------

1) Install the `Darknet` framework, I recommend the following repository
https://github.com/hank-ai/darknet
2) Clone https://github.com/LeonardoFaggiani/JustAnotherAlpr.git
3) Go to the darknet folder and modify the darknet.py file on line 328 "{path/to/your}/darknet.dll" replace {path/to/your} with the path where the darknet installer is located, save it.
4) Go to the nn/license-plate path and modify the license-plate.data file and you must tell darknet where your license-plate.names file is, for example if you cloned the JustAnotherAlpr repository on the C drive, it would be C://JustAnotherAlpr/nn/license-plate/license-plate.names

Usage
-----

## Video/Webcam/Images

### Process a Video File
    python darknet_video.py --input samples\Testing-Driving.mp4 --weights ..\nn\license-plate\license-plate.weights --config_file ..\nn\license-plate\license-plate.cfg --data_file ..\nn\license-plate\license-plate.data

### Use Display for Real-Time Detection
    python darknet_video.py --input 0 --weights ..\nn\license-plate\license-plate.weights --config_file ..\nn\license-plate\license-plate.cfg --data_file ..\nn\license-plate\license-plate.data

### Use Webcam for Real-Time Detection
    python darknet_video.py --input 1 --weights ..\nn\license-plate\license-plate.weights --config_file ..\nn\license-plate\license-plate.cfg --data_file ..\nn\license-plate\license-plate.data

### Save Processed Video
    python darknet_video.py --input samples\Testing-Driving.mp4 --out_filename processed_Testing-Driving.avi --weights ..\nn\license-plate\license-plate.weights --config_file ..\nn\license-plate\license-plate.cfg --data_file ..\nn\license-plate\license-plate.data

### Single Image Detection
    python darknet_images.py --input samples\Testing-Image.jpg --weights ..\nn\license-plate\license-plate.weights --config_file ..\nn\license-plate\license-plate.cfg --data_file ..\nn\license-plate\license-plate.data --gpu

License Plate Metrics
-----
![image](https://github.com/user-attachments/assets/f25cf85f-5068-4334-8e01-01c2f443832c)

![image](https://github.com/user-attachments/assets/10ce5c59-1d5c-4b39-89e9-5d13f6af41c6)



## TODO

* [x] Upload differents kind of images to train the model.
* [x] Create Model
* [x] Draw boxes in video streaming
* [ ] OCR 🛠️ (In Progress)
* [ ] Information of the owner of the vehicle license (Simulation)
* [ ] Move images to blob storage 