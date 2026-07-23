# ============================================================
# recognizers/holistic_recognizer.py
# ============================================================
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python.vision import (RunningMode, HolisticLandmarker, HolisticLandmarkerOptions)
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarksConnections
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
from recognizers.base import BaseRecognizer

class HolisticRecognizer(BaseRecognizer):
    def _model_file(self) -> str:
        return "holistic_landmarker.task"

    def _create_task(self):
        opts = HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.IMAGE,
            min_face_detection_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_hand_landmarks_confidence=0.5)
        return HolisticLandmarker.create_from_options(opts)

    def process(self, frame, mp_image):
        result = self._task.detect(mp_image)
        h, w = frame.shape[:2]
        if result.pose_landmarks:
            for lm_list in result.pose_landmarks:
                for conn in PoseLandmarksConnections.POSE_LANDMARKS:
                    s, e = lm_list[conn.start], lm_list[conn.end]
                    cv.line(frame, (int(s.x*w), int(s.y*h)), (int(e.x*w), int(e.y*h)), (100,255,100), 2)
                for lm in lm_list:
                    cv.circle(frame, (int(lm.x*w), int(lm.y*h)), 2, (0,0,255), -1)
        if result.face_landmarks:
            for lm_list in result.face_landmarks:
                for conn in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
                    s, e = lm_list[conn.start], lm_list[conn.end]
                    cv.line(frame, (int(s.x*w), int(s.y*h)), (int(e.x*w), int(e.y*h)), (0,255,200), 1)
        if result.left_hand_landmarks:
            for lm_list in result.left_hand_landmarks:
                for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                    s, e = lm_list[conn.start], lm_list[conn.end]
                    cv.line(frame, (int(s.x*w), int(s.y*h)), (int(e.x*w), int(e.y*h)), (255,150,0), 1)
                for lm in lm_list:
                    cv.circle(frame, (int(lm.x*w), int(lm.y*h)), 2, (0,0,255), -1)
        if result.right_hand_landmarks:
            for lm_list in result.right_hand_landmarks:
                for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                    s, e = lm_list[conn.start], lm_list[conn.end]
                    cv.line(frame, (int(s.x*w), int(s.y*h)), (int(e.x*w), int(e.y*h)), (0,200,255), 1)
                for lm in lm_list:
                    cv.circle(frame, (int(lm.x*w), int(lm.y*h)), 2, (0,0,255), -1)
        return frame