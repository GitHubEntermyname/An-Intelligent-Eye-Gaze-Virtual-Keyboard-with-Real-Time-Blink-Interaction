import cv2

class Pupil(object):
    def __init__(self, eye_frame, calibration, threshold=70):
        """
        Initialize pupil detection.
        :param eye_frame: cropped frame of the eye
        :param calibration: calibration object (used later for scaling)
        :param threshold: integer threshold for binary segmentation
        """
        self.iris_frame = None
        self.x = None
        self.y = None
        self.calibration = calibration
        self.threshold = int(threshold)  # ✅ Always numeric

        self.detect_iris(eye_frame)

    def image_processing(self, frame, threshold):
        """
        Apply thresholding to isolate the iris.
        """
        if len(frame.shape) == 3:  # Convert to grayscale if needed
            new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            new_frame = frame

        _, new_frame = cv2.threshold(new_frame, int(threshold), 255, cv2.THRESH_BINARY)
        return new_frame

    def detect_iris(self, eye_frame):
        """
        Detect the iris region.
        """
        self.iris_frame = self.image_processing(eye_frame, self.threshold)
        # TODO: Add contour detection & centroid for (self.x, self.y)
        self.x, self.y = None, None
