from recognizers.base import BaseRecognizer
from recognizers.face_detect import FaceDetectionRecognizer
from recognizers.face_mesh import FaceMeshRecognizer
from recognizers.gesture import GestureRecognizer_
from recognizers.hair_seg import HairSegmentationRecognizer
from recognizers.hand_recog import HandRecognizer
from recognizers.holistic import HolisticRecognizer
from recognizers.iris_eye import IrisTrackingRecognizer
from recognizers.pose import PoseRecognizer
from recognizers.selfie_seg import SelfieSegmentationRecognizer
MODE_REGISTRY = {
    1: ("Hand", "", HandRecognizer),
    2: ("Face Detection", "", FaceDetectionRecognizer),
    3: ("Face Mesh", "", FaceMeshRecognizer),
    4: ("Selfie Segmentation", "", SelfieSegmentationRecognizer),
    5: ("Gesture", "", GestureRecognizer_),
    6: ("Pose", "", PoseRecognizer),
    7: ("Holistic", "", HolisticRecognizer),
    8: ("Hair Segmentation", "", HairSegmentationRecognizer),
    9: ("Iris Tracking", "", IrisTrackingRecognizer),
}
