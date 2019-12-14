import cv2
import os
import numpy as np
from hand_tracker import HandTracker
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from collections import Counter

K.clear_session()

mdl = load_model('trained_400.h5')


WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "./palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "./hand_landmark.tflite"
ANCHORS_PATH = "./anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)

if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

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
predict = []
font = cv2.FONT_HERSHEY_SIMPLEX

k=[]
present =''
while hasFrame:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)
    if points is not None:
        inp = np.array(points)
        inp[:,0] = np.divide(np.subtract(inp[:,0],inp[:,0].mean()),inp[:,0].std())
        inp[:,1] = np.divide(np.subtract(inp[:,1],inp[:,1].mean()),inp[:,1].std())
        inp = inp.reshape(1,inp.shape[0]*inp.shape[1])
        pred = mdl.predict(inp)
        val = np.argmax(pred[0])
        # print("Predicted values is :"+chr(65+val))
        if(len(predict)<30):
            predict.append(chr(65+val))
        if(len(predict)==30):
            del predict[0]
            predict.append(chr(65+val))
            a = Counter(predict)
            next_char = list((a.most_common())[0])
            count = next_char[1]
            next_char = next_char[0]
            if(count>=20):
                if(next_char != present):
                    k.append(next_char)
                    present = next_char
        for point in points:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
    strr = ''.join(k)
    cv2.putText(frame, strr, (10, 100), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow(WINDOW, frame)
    hasFrame, frame = capture.read()
    key = cv2.waitKey(20)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
