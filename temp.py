## Important :
## directory strucrutre should be as follows
## root -> run0.py
#         asl_alphabet_train --> asl_alphabet_train --> A
##                                                      B
##                                                      .

import cv2
import os
import numpy as np
from hand_tracker import HandTracker
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

K.clear_session()

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "./palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "./hand_landmark.tflite"
ANCHORS_PATH = "./anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity

# cv2.namedWindow(WINDOW)
# capture = cv2.VideoCapture(0)
#
# if capture.isOpened():
#     hasFrame, frame = capture.read()
# else:
#     hasFrame = False

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)
dir = 'asl_alphabet_test'
mdl = load_model('trained_400.h5')
c = os.listdir(dir)
count = 0

total=0
for j in c:
    d = os.listdir(dir+'/'+j)
    for i in d:
        total = total +1
        frame = cv2.imread(dir+'/'+j+'/'+i,1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Some Image Pre-processing
        yen_threshold = threshold_yen(image)
        image = rescale_intensity(image, (0, yen_threshold), (0, 255))

        points, _ = detector(image)
        if points is not None:
            inp = np.array(points)
            inp[:,0] = np.divide(np.subtract(inp[:,0],inp[:,0].mean()),inp[:,0].std())
            inp[:,1] = np.divide(np.subtract(inp[:,1],inp[:,1].mean()),inp[:,1].std())
            inp = inp.reshape(1,inp.shape[0]*inp.shape[1])
            pred = mdl.predict(inp)
            val = np.argmax(pred[0])
            print("Predicted values is :"+chr(65+val)+" and the truth value is : "+str(i[0]))

            for point in points:
                x, y = point
                cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            for connection in connections:
                x0, y0 = points[connection[0]]
                x1, y1 = points[connection[1]]
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
