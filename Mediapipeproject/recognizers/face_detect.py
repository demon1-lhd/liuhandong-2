# ============================================================
# recognizers/face_detection_recognizer.py
# ============================================================
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python.vision import (RunningMode, FaceDetector, FaceDetectorOptions)
from recognizers.base import BaseRecognizer

class FaceDetectionRecognizer(BaseRecognizer):
    def _model_file(self) -> str:
        return "blaze_face_short_range.tflite"

    def _create_task(self):
        opts = FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.IMAGE, min_detection_confidence=0.5)
        return FaceDetector.create_from_options(opts)

    def process(self, frame, mp_image):
        result = self._task.detect(mp_image)
        if not result.detections:
            return frame
        for det in result.detections:
            bb = det.bounding_box
            if bb:
                cv.rectangle(frame, (bb.origin_x, bb.origin_y),
                             (bb.origin_x+bb.width, bb.origin_y+bb.height), (0,255,0), 2)
            if det.keypoints:
                for kp in det.keypoints:
                    cv.circle(frame, (int(kp.x), int(kp.y)), 3, (0,0,255), -1)
        return frame