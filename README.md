# YOLOX-DeepSort-ANPR

![tokyo](assets/tokyo.gif)

## Overview

ANPR utilizing YOLOX object detection from a custom trained dataset to detect license plates, and deepsort object tracking. The detection frames then pipelined to Tesseract OCR to detect characters (TODO)

## How to Use

### YOLOX

Don't forget to install yolox first, guide can be found on the official YOLOX repository:
[Megvii-BaseDetection YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
or you can just simply
`python3 ./setup.py`

### PyTesseract

Detailed PyTesseract installation guide can be read here:
[www.projectpro.io, Install PyTesseract](https://www.projectpro.io/recipes/what-is-pytesseract-python-library-and-do-you-install-it)

### Example

to run detection on a video:
`python3 ./demo.py --path ./path/to/video`

## Datasets Used

Original dataset (VOC2017 format) that is used can be found at
[andrewmd - Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download)
However, training is done on COCO JSON Format dataset that could be found here:
[COCO ANPR - Google Drive](https://drive.google.com/file/d/1_R9f0N-u6nVM4Wx3LBAmiH5ougjtoi08/view?usp=sharing)

## Future Improvements

- Optimize model accuracy
- Restructure project
- Optimize engine efficiency
