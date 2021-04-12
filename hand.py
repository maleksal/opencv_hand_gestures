import cv2

from modules.Gestures import Gesture
from modules.HandDetection import HandDetection

cap = cv2.VideoCapture(0)
detector = HandDetection(detection_confidence=0.8,
                         tracking_confidence=0.6)
gestures = Gesture()

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
    img = detector.process_hand(img)
    lmlist = detector.hand_coordinates(img)
    if len(lmlist) > 0:
        gestures.start(lmlist, img)
    cv2.imshow('test module', img)
    # Close cam
    cv2.waitKey(1)
