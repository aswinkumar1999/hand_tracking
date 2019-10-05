## Hand tracking

Forked from : https://github.com/metalwhale/hand_tracking

### 1. File Description
- `palm_detection_without_custom_op.tflite` - For the Palm Detection with the custom operations
- `hand_landmark.tflite` - For the Landmark detection 
- `anchors.csv` and `hand_tracker.py` for running those code.

### 2. Setup 
```
$ git clone https://github.com/aswinkumar1999/hand_tracking.git
$ pip3 install opencv-python tensorflow
$ python3 run.py
```

### 3. Other Useful Resources
[mediapipe-models](https://github.com/junhwanjang/mediapipe-models/tree/master/palm_detection/mediapipe_models)

[mediapipe](https://github.com/google/mediapipe/tree/master/mediapipe/models)

[hand_tracking](https://github.com/wolterlw/hand_tracking)

### 4. Output
![Result](/output.gif?raw=true "Result")
