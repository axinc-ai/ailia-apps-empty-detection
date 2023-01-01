# ailia APPS Empty Detection

Detects cars in parking lots, inventory on shelves, etc., and counts vacant information.

## Functions

- Empty area detection
- Export count to csv

## Requirements

- Windows, macOS, Linux
- Python 3.7 and later
- [ailia SDK](https://github.com/axinc-ai/ailia-models/blob/master/TUTORIAL.md) 1.2.14 and later

## Basic Usage

1. Put this command to open GUI.

```
python3 ailia-apps-empty-detection.py
```

![Open GUI](./tutorial/open.jpg)

2. Push "Input video" button to select input video
3. Push "Set area" button to set area

![Set area](./tutorial/area.jpg)

Click on the screen to draw two lines.

4. Push "Run" button to execute the app

![Run app](./tutorial/run.jpg)

## Other functions

### Write output to video and csv

a. Push "Output video" button to select output video
b. Push "Output csv" button to select output csv

The examples of csv file.

```
time(sec) , area0 , area1
0 , 0 , 0
1 , 0 , 1
2 , 1 , 0
```

## Architecture

```mermaid
classDiagram
`ailia APPS Empty Detection` <|-- `Detic` : Empty area detection (area matching)
`Detic` <|-- `ailia.core` : Inference result
`ailia APPS Empty Detection` : Input Video, Output Video, Output csv
`Detic` : Large object segmentation
`ailia.core` <|-- `onnx` : Model
`ailia.core` : ailiaCreate, ailiaPredict, ailiaDestroy
`ailia.core` <|-- Backend : Acceleration
`onnx` : detic
`Backend` : CPU, GPU
```

## Test video

https://pixabay.com/videos/building-parking-lot-parking-car-130571/
