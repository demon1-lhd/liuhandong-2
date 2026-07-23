# ============================================================
# recognizers/gesture_recognizer.py
# ============================================================
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python.vision import (RunningMode, GestureRecognizer, GestureRecognizerOptions)
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from recognizers.base import BaseRecognizer

class GestureRecognizer_(BaseRecognizer):
    def _model_file(self) -> str:
        return "gesture_recognizer.task"

    def _create_task(self):
        opts = GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
        return GestureRecognizer.create_from_options(opts)

    def process(self, frame, mp_image):
        result = self._task.recognize(mp_image)
        if not result.hand_landmarks:
            return frame
        h, w = frame.shape[:2]
        for i, lm_list in enumerate(result.hand_landmarks):
            color = (0, 255, 0)
            if result.handedness and i < len(result.handedness):
                cat = result.handedness[i][0].category_name
                color = (255, 100, 0) if cat == "Left" else (0, 200, 255)
            for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                s, e = lm_list[conn.start], lm_list[conn.end]
                cv.line(frame, (int(s.x*w), int(s.y*h)), (int(e.x*w), int(e.y*h)), color, 1)
            for lm in lm_list:
                cv.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0, 0, 255), -1)
            gesture = ""
            if result.gestures and i < len(result.gestures) and result.gestures[i]:
                gesture = result.gestures[i][0].category_name
            if gesture:
                idx0 = lm_list[0]
                tx, ty = int(idx0.x*w)-30, int(idx0.y*h)-20
                cv.putText(frame, gesture, (max(tx,0), max(ty,15)),
                           cv.FONT_HERSHEY_COMPLEX, 0.9, (255,255,0), 2)
        return frame