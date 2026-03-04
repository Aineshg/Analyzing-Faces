from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import cv2
import face_recognition
import numpy as np

from analyzing_faces.models.face_recognizer import FaceRecognizer
from analyzing_faces.models.mask_detector import MaskDetector
from analyzing_faces.services.attendance import AttendanceEvent, AttendanceService


@dataclass(slots=True)
class FrameDetection:
    person_name: str
    confidence: float
    mask_label: str
    mask_probability: float
    location: tuple[int, int, int, int]


class RealTimeAnalyzer:
    def __init__(
        self,
        face_recognizer: FaceRecognizer,
        mask_detector: MaskDetector,
        attendance_service: AttendanceService,
    ) -> None:
        self.face_recognizer = face_recognizer
        self.mask_detector = mask_detector
        self.attendance_service = attendance_service

    def analyze_frame(self, frame_bgr: np.ndarray) -> list[FrameDetection]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")

        detections: list[FrameDetection] = []
        for top, right, bottom, left in locations:
            match = self.face_recognizer.identify(rgb, (top, right, bottom, left))
            face_crop = frame_bgr[top:bottom, left:right]
            if face_crop.size == 0:
                continue

            mask = self.mask_detector.predict(face_crop)
            detections.append(
                FrameDetection(
                    person_name=match.name,
                    confidence=match.confidence,
                    mask_label=mask.label,
                    mask_probability=mask.probability,
                    location=(top, right, bottom, left),
                )
            )

            if match.name != "UNKNOWN":
                self.attendance_service.record_once_per_day(
                    AttendanceEvent(
                        person_name=match.name,
                        timestamp=datetime.utcnow(),
                        confidence=match.confidence,
                        mask_label=mask.label,
                    )
                )
        return detections
