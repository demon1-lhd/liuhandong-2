import sys, os, time
import cv2 as cv
import mediapipe as mp
from PySide6.QtWidgets import QApplication, QPushButton, QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer
from PySide6.QtUiTools import QUiLoader
from recognizers import MODE_REGISTRY

BUTTON_TO_MODE = {
    "faceButton": 2, "facewangButton": 3, "photoButton": 4,
    "gestureButton": 5, "poseButton": 6, "bodyButton": 7,
    "hairButton": 8, "eyeButton": 9,
}

BTN_NAMES = ["faceButton","facewangButton","photoButton","gestureButton",
             "poseButton","bodyButton","hairButton","eyeButton"]

class MediaPipeApp:
    def __init__(self):
        self._cap = None
        self._timer = None
        self._recognizer = None
        self._current_mode = None
        self._camera_on = False
        self._fps_time = time.time()
        self._fps_count = 0
        self._fps_display = 0.0

        loader = QUiLoader()
        self._window = loader.load("test.ui")
        self._title_label = self._window.findChild(QLabel, "titleLabel")
        self._cap_label   = self._window.findChild(QLabel, "capLabel")
        self._open_btn    = self._window.findChild(QPushButton, "openButton")
        self._stop_btn    = self._window.findChild(QPushButton, "stopButton")

        self._open_btn.clicked.connect(self._on_open)
        self._stop_btn.clicked.connect(self._on_stop)

        for name in BTN_NAMES:
            btn = self._window.findChild(QPushButton, name)
            if btn:
                mid = BUTTON_TO_MODE[name]
                btn.clicked.connect(self._make_callback(mid))

        self._cap_label.setText("请点击功能按钮开始")
        self._cap_label.setAlignment(Qt.AlignCenter)
        self._cap_label.setStyleSheet("background-color:#1a1a2e; color:#888; font-size:18px;")

    def _make_callback(self, mode_id):
        return lambda: self._on_feature(mode_id)

    def _on_open(self):
        if self._camera_on: self._stop_camera()
        else:
            if self._current_mode is None: self._current_mode = 2
            self._start_camera()

    def _on_feature(self, mode_id):
        print(f"[BTN] mode={mode_id}")
        if mode_id == 7:
            self._cap_label.setText("全身检测在此版本暂不可用\n(MediaPipe 0.10.21 Holistic C++ bug)")
            return
        if self._current_mode == mode_id and self._camera_on: return
        self._current_mode = mode_id
        if self._camera_on: self._stop_camera()
        self._start_camera()

    def _on_stop(self):
        self._stop_camera()
        self._window.close()

    def _start_camera(self):
        if self._current_mode is None: return
        self._unload_recognizer()
        _, _, RCls = MODE_REGISTRY[self._current_mode]
        rec = RCls()
        if not os.path.exists(rec.model_path):
            self._cap_label.setText(f"缺少模型文件:\n{rec.model_path}"); return
        if not rec.load():
            self._cap_label.setText(f"模型加载失败:\n{rec.model_path}"); return
        self._recognizer = rec
        self._cap = cv.VideoCapture(0)
        if not self._cap.isOpened():
            self._cap_label.setText("无法打开摄像头")
            self._cap = None; self._unload_recognizer(); return
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_frame)
        self._timer.start(10)
        self._camera_on = True
        name, _, _ = MODE_REGISTRY[self._current_mode]
        self._title_label.setText(f"MediaPipe - {name} ")

    def _stop_camera(self):
        if self._timer: self._timer.stop(); self._timer = None
        if self._cap: self._cap.release(); self._cap = None
        self._unload_recognizer()
        self._camera_on = False
        self._cap_label.setText("请点击功能按钮开始")
        self._cap_label.setStyleSheet("background-color:#1a1a2e; color:#888; font-size:18px;")
        self._title_label.setText("Mediapipe")

    def _unload_recognizer(self):
        if self._recognizer: self._recognizer.unload(); self._recognizer = None

    def _on_frame(self):
        if not self._cap or not self._cap.isOpened() or not self._recognizer: return
        ret, frame = self._cap.read()
        if not ret: return
        frame = cv.flip(frame, 1)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame = self._recognizer.process(frame, mp_img)
        self._fps_count += 1
        now = time.time()
        if now - self._fps_time >= 1.0:
            self._fps_display = self._fps_count / (now - self._fps_time)
            self._fps_count = 0; self._fps_time = now
        cv.putText(frame, f"FPS:{self._fps_display:.0f}",
                   (frame.shape[1]-90, frame.shape[0]-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        lw, lh = self._cap_label.width(), self._cap_label.height()
        if lw > 0 and lh > 0: frame = cv.resize(frame, (lw, lh))
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, 3*w, QImage.Format_RGB888).rgbSwapped()
        self._cap_label.setPixmap(QPixmap.fromImage(qimg))

    def run(self):
        self._window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MediaPipeApp().run()
    sys.exit(app.exec())