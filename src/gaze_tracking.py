import cv2
from .pupil import Pupil

class GazeTracking(object):
    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = None

    def refresh(self, frame):
        """Update the frame and analyze gaze"""
        self.frame = frame
        self._analyze()

    def _analyze(self):
        """Detect pupils"""
        height, width = self.frame.shape[:2]
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        left_frame = gray[0:height, 0:width // 2]
        right_frame = gray[0:height, width // 2:width]

        self.eye_left = Pupil(left_frame, self.calibration)
        self.eye_right = Pupil(right_frame, self.calibration)

    def is_blinking(self):
        return self.eye_left.x is None or self.eye_right.x is None

    def is_right(self):
        if self.eye_left.x is not None and self.eye_right.x is not None:
            return self.eye_left.x < self.eye_right.x
        return False

    def is_left(self):
        if self.eye_left.x is not None and self.eye_right.x is not None:
            return self.eye_left.x > self.eye_right.x
        return False

    def is_up(self):
        if self.eye_left.y is not None and self.eye_right.y is not None:
            return self.eye_left.y < self.eye_right.y
        return False

    def is_down(self):
        if self.eye_left.y is not None and self.eye_right.y is not None:
            return self.eye_left.y > self.eye_right.y
        return False
