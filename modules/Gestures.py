"""
Gesture detection Module
"""
import math
import sys

import cv2
import numpy as np


class Gesture:
    """Responsible for gesture detection."""

    __FingerTips = [4, 8, 12, 16, 20]

    def __init__(self):
        """Initializer.
        """
        self.coordinates, self.img = None, None
        self.call_function = {
            "11000": self.volume_control,
        }
        self.volume_controller = None
        if sys.platform.startswith('win'):
            # To use specific lib for audio control on windows.
            from modules.VolumeController import Controller
            self.volume_controller = Controller()

    def extract_pattern(self, coordinates):
        """
        Extract finger patterns (finger up or down) from hand coordinates
        :return list of zeros and ones.
        """
        pattern = []
        # Check thumb
        if coordinates[self.__FingerTips[0]][1] > \
                coordinates[self.__FingerTips[0] - 1][1]:
            pattern.append(1)
        else:
            pattern.append(0)
        # Check remaining 4 fingers
        for ft in range(1, 5):
            if coordinates[self.__FingerTips[ft]][2] < \
                    coordinates[self.__FingerTips[ft] - 2][2]:
                pattern.append(1)
            else:
                pattern.append(0)
        return "".join(str(_) for _ in pattern)

    def volume_control(self, draw=True):
        """Controls the volume.
        :param draw: if True, draw graphics to image.
        """
        x1, y1 = self.coordinates[4][1], self.coordinates[4][2]
        x2, y2 = self.coordinates[8][1], self.coordinates[8][2]
        # center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # distance between points
        length = math.hypot(x2 - x1, y2 - y1)

        if self.volume_controller:
            # Hand range 50 - 160
            self.volume_controller.set_volume(length, [50, 160])
        if draw:
            test_max_volume = 150
            test_min_volume = 400
            # vol here used for graphics.
            vol = np.interp(length, [50, 160], [test_min_volume, test_max_volume])
            volper = np.interp(length, [50, 160], [0, 100])
            # draw circles
            cv2.circle(self.img, (x1, y1), 13, (255, 0, 255), cv2.FILLED)
            cv2.circle(self.img, (x2, y2), 13, (255, 0, 255), cv2.FILLED)
            cv2.line(self.img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(self.img, (cx, cy), 13, (255, 0, 255), cv2.FILLED)
            # graphic volume bar
            cv2.rectangle(self.img, (50, 150), (80, 400), (0, 255, 0))
            cv2.rectangle(self.img, (50, int(vol)), (80, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(self.img, f'{int(volper)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                        1, (255, 0, 0), 3)

    def start(self, finger_coordinates, image):
        """Calls the right function after analysing finger_coordinates.
        :param finger_coordinates:  (x, y) coordinates of hand fingers in pixels
        :param image:               image from cap.read().
        """
        # initialize values
        self.img, self.coordinates = image, finger_coordinates

        pattern = self.extract_pattern(finger_coordinates)
        if pattern in self.call_function.keys():
            self.call_function[pattern]()
