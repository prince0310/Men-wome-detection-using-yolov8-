# Men-wome-detection-using-yolov8

#### This guide will provide instructions on how to convert OIDv4 data into the YOLO format for use with YOLOv4 object detection algorithms.

#### Getting Started
``` git clone https://github.com/prince0310/Men-wome-detection-using-yolov8-.git ```

<details open>
<summary>Dataset</summary>
  <br>
  For training custom data set on yolo model you need to have data set arrangement in yolo format. which includes Images and Their annotation file.
  
  ```                
Custom dataset
        |
        |─── train
        |    |
        |    └───Images --- 0fdea8a716155a8e.jpg
        |    └───Labels --- 0fdea8a716155a8e.txt
        |
        └─── test
        |    └───Images --- 0b6f22bf3b586889.jpg
        |    └───Labels --- 0b6f22bf3b586889.txt
        |
        └─── validation
        |    └───Images --- 0fdea8a716155a8e.jpg
        |    └───Labels --- 0fdea8a716155a8e.txt
        |
        └─── data.yaml
```
  
</details>


<details open>
<summary>Install</summary>
 
Pip install the ultralytics package including
all [requirements.txt](https://github.com/ultralytics/ultralytics/blob/main/requirements.txt) in a
[**3.10>=Python>=3.7**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).
  
```bash
pip install ultralytics
```
</details>

<details open>
<summary>Train</summary>
  <br>
  
Python 
  
```bash
from ultralytics import YOLO

# Train
model = YOLO("yolov8n.pt")

results = model.train(data="data.yaml", epochs=200, workers=1, batch=8,imgsz=640)  # train the model
```
Cli
  
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=200 imgsz=640
  ```
</details>

<details open>
<summary>Detect</summary>
   <br>
  
  Python 
  
```bash
from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # load a custom model

# Predict with the model
results = model("image.jpg", save = True)  # predict on an image
```
Cli
  
```bash
yolo detect predict model=path/to/best.pt source="images.jpg"  # predict with custom model
  ```
  
</details>
