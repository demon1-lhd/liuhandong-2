#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import cv2 as cv
import mediapipe as mp
class BaseRecognizer(ABC):
    def __init__(self):
        self._task = None
        self._model_path: str = self._model_file()
    @abstractmethod
    def _model_file(self) -> str:
        ...
    @abstractmethod
    def _create_task(self):
        ...
    @abstractmethod
    def process(self, frame: cv.Mat, mp_image: mp.Image) -> cv.Mat:
        ...
    @property
    def model_path(self) -> str:
        return self._model_path
    def load(self) -> bool:
        import os
        if not os.path.exists(self._model_path):
            return False
        try:
            self._task = self._create_task()
            return True
        except Exception:
            return False
    def unload(self):
        if self._task is not None:
            try:
                self._task.close()
            except Exception:
                pass
            self._task = None
    @property
    def is_loaded(self) -> bool:
        return self._task is not None
