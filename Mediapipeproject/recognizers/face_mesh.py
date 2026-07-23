# ============================================================
# recognizers/face_mesh_recognizer.py
# ============================================================
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks.python.vision import (RunningMode, FaceLandmarker, FaceLandmarkerOptions)
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
from recognizers.base import BaseRecognizer

FT = FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
FC = FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
FLI = FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS
FRI = FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS

class FaceMeshRecognizer(BaseRecognizer):
    def _model_file(self) -> str:
        return "face_landmarker.task"

    def _create_task(self):
        opts = FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.IMAGE, num_faces=1,
            output_face_blendshapes=False, output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5, min_tracking_confidence=0.5)
        return FaceLandmarker.create_from_options(opts)

    def process(self, frame, mp_image):
        result = self._task.detect(mp_image)
        if not result.face_landmarks:
            return frame
        for lm_list in result.face_landmarks:
            self._draw(frame, lm_list, FT, (180,220,180), 1, 0)
            self._draw(frame, lm_list, FC, (0,255,0), 1, 1)
            self._draw(frame, lm_list, FLI, (0,255,255), 1, 2)
            self._draw(frame, lm_list, FRI, (0,255,255), 1, 2)
        return frame

    def _draw(self, img, lm, conns, color, thick, radius):
        h, w = img.shape[:2]
        for c in conns:
            s, e = lm[c.start], lm[c.end]
            cv.line(img, (int(s.x*w), int(s.y*h)), (int(e.x*w), int(e.y*h)), color, thick)
        if radius > 0:
            for l in lm:
                cv.circle(img, (int(l.x*w), int(l.y*h)), radius, color, -1)