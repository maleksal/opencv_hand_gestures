"""
Hand Detection Module
"""
import cv2
import mediapipe as mp


class HandDetection:
    """
    Responsible for hand detection using mediapipe.
    """

    __RESULTS = []

    def __init__(self,
                 static_mode=False,
                 max_hands=2,
                 detection_confidence=0.5,
                 tracking_confidence=0.5):
        """Initializes.

        :param static_mode: Whether to treat the input images as a batch of static
              and possibly unrelated images, or a video stream.
        :param max_hands: Maximum number of hands to detect.
        :param detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand
              detection to be considered successful.
        :param tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
              hand landmarks to be considered tracked successfully.
        """

        self.static_mode = static_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.mphands = mp.solutions.hands
        self.detection = self.mphands.Hands(self.static_mode, self.max_hands,
                                            self.detection_confidence, self.tracking_confidence)
        self.detection_drawing = mp.solutions.drawing_utils

    def process_hand(self, image, draw=True):
        """Process an image and detects hands
        :param image: image from cap.read().
        :param draw:  draw hands landmarks on image.
        """

        # the BGR image to RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image_rgb.flags.writeable = False
        self.__RESULTS = self.detection.process(image_rgb)

        if self.__RESULTS.multi_hand_landmarks and draw:
            image.flags.writeable = True
            for handLms in self.__RESULTS.multi_hand_landmarks:
                self.detection_drawing.draw_landmarks(image, handLms,
                                                      self.mphands.HAND_CONNECTIONS)
        return image

    def hand_coordinates(self, image, max_hand=0, draw=False):
        """Returns coordinates of detected hands in pixels
        :param image:     image from cap.read().
        :param max_hand:  number of hands to extract.
        :param draw:      if true, draw circle in each coordinate.
        """
        landmarks = []
        calculate = (lambda h, w, x, y: (x * w, y * h))

        if self.__RESULTS.multi_hand_landmarks:
            hand = self.__RESULTS.multi_hand_landmarks[max_hand]
            height, width, c = image.shape  # get image height && width
            for lid, landmark in enumerate(hand.landmark):
                cx, cy = calculate(height, width, landmark.x, landmark.y)
                landmarks.append([lid, int(cx), int(cy)])
                if draw:
                    cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return landmarks
