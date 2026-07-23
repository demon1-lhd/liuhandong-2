# ============================================================
# recognizers/iris_tracking_recognizer.py
# ============================================================
import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import (RunningMode, FaceLandmarker, FaceLandmarkerOptions)
from recognizers.base import BaseRecognizer

RID = 11.7
FL = 850.0
LI = [469,470,471,472]
RI = [474,475,476,477]
LE = [33,133,160,144,158,153,154,155]
RE = [362,263,387,373,385,380,381,382]

class IrisTrackingRecognizer(BaseRecognizer):
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
        h, w = frame.shape[:2]
        result = self._task.detect(mp_image)
        if not result.face_landmarks:
            return frame
        lm = result.face_landmarks[0]
        ld = self._dia(lm, w, h, LI)
        rd = self._dia(lm, w, h, RI)
        dists = [d for d in (self._dist(ld), self._dist(rd)) if d]
        avg = np.mean(dists) if dists else 0
        le = self._ear(lm, w, h, LE)
        re = self._ear(lm, w, h, RE)
        for idx in LI + RI:
            cv.circle(frame, (int(lm[idx].x*w), int(lm[idx].y*h)), 2, (0,255,255), -1)
        for idx in (468,473):
            cv.circle(frame, (int(lm[idx].x*w), int(lm[idx].y*h)), 3, (0,255,0), -1)
        for indices in (LE, RE):
            pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in indices]
            for i in range(len(pts)):
                cv.line(frame, pts[i], pts[(i+1)%len(pts)], (255,0,0), 1)
        cv.putText(frame, f"Distance: {avg:.0f} mm", (10,30),
                   cv.FONT_HERSHEY_COMPLEX, 0.9, (0,255,0), 2)
        cv.putText(frame, f"L-eye: {le:.2f}  R-eye: {re:.2f}", (10,60),
                   cv.FONT_HERSHEY_COMPLEX, 0.6, (200,200,200), 1)
        return frame

    def _dia(self, lm, w, h, indices):
        pts = np.array([[lm[i].x*w, lm[i].y*h] for i in indices])
        return float(np.linalg.norm(pts[1]-pts[3]))

    def _dist(self, d):
        return (FL*RID)/d if d > 0 else None

    def _ear(self, lm, w, h, indices):
        pts = np.array([[lm[i].x*w, lm[i].y*h] for i in indices])
        hy = pts[:,1].max()-pts[:,1].min()
        wx = pts[:,0].max()-pts[:,0].min()
        return hy/wx if wx > 0 else 0