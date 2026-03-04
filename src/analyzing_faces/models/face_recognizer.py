from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import face_recognition
import numpy as np


@dataclass(slots=True)
class IdentityMatch:
    name: str
    confidence: float


class FaceRecognizer:
    def __init__(self, known_faces_dir: Path, threshold: float = 0.52) -> None:
        self.threshold = threshold
        self._labels: list[str] = []
        self._encodings: list[np.ndarray] = []
        self._bootstrap_known_faces(known_faces_dir)

    def _bootstrap_known_faces(self, known_faces_dir: Path) -> None:
        if not known_faces_dir.exists():
            return

        for person_dir in known_faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
            for image_path in person_dir.glob("*.*"):
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self._labels.append(person_dir.name)
                    self._encodings.append(encodings[0])

    def identify(
        self, rgb_frame: np.ndarray, face_location: tuple[int, int, int, int]
    ) -> IdentityMatch:
        if not self._encodings:
            return IdentityMatch(name="UNKNOWN", confidence=0.0)

        encoding = face_recognition.face_encodings(rgb_frame, [face_location])
        if not encoding:
            return IdentityMatch(name="UNKNOWN", confidence=0.0)

        distances = face_recognition.face_distance(self._encodings, encoding[0])
        min_idx = int(np.argmin(distances))
        min_distance = float(distances[min_idx])

        if min_distance > self.threshold:
            return IdentityMatch(name="UNKNOWN", confidence=max(0.0, 1.0 - min_distance))

        return IdentityMatch(name=self._labels[min_idx], confidence=max(0.0, 1.0 - min_distance))
