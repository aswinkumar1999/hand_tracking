import cv2
import os
from hand_tracker import HandTracker

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

c = os.listdir('img')
count = 0
for i in c:
    frame = cv2.imread('img'+'/'+i,1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Some Image Pre-processing
    yen_threshold = threshold_yen(image)
    image = rescale_intensity(image, (0, yen_threshold), (0, 255))

    points, _ = detector(image)
    if points is not None:
        count = count+1
        for point in points:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
    cv2.imshow('Output', frame)
    cv2.waitKey(0)

print(count)
cv2.destroyAllWindows()
