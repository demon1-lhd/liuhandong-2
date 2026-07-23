# ============================================================
# recognizers/pose_recognizer.py
# ============================================================
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python.vision import (RunningMode, PoseLandmarker, PoseLandmarkerOptions)
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections
from recognizers.base import BaseRecognizer

class PoseRecognizer(BaseRecognizer):
    def _model_file(self) -> str:
        return "pose_landmarker.task"

    def _create_task(self):
        opts = PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5)
        return PoseLandmarker.create_from_options(opts)

    def process(self, frame, mp_image):
        result = self._task.detect(mp_image)
        if not result.pose_landmarks:
            return frame
        h, w = frame.shape[:2]
        for lm_list in result.pose_landmarks:
            for conn in PoseLandmarksConnections.POSE_LANDMARKS:
                s, e = lm_list[conn.start], lm_list[conn.end]
                cv.line(frame, (int(s.x*w), int(s.y*h)), (int(e.x*w), int(e.y*h)), (0,255,100), 2)
            for lm in lm_list:
                cv.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0,0,255), -1)
        return frame