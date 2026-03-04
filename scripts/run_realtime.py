from __future__ import annotations

import argparse

import cv2

from analyzing_faces.config.settings import settings
from analyzing_faces.models.face_recognizer import FaceRecognizer
from analyzing_faces.models.mask_detector import MaskDetector
from analyzing_faces.pipelines.realtime_analyzer import RealTimeAnalyzer
from analyzing_faces.services.attendance import AttendanceService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run realtime face analysis pipeline")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    attendance = AttendanceService(settings.attendance_file)
    recognizer = FaceRecognizer(settings.known_faces_dir, threshold=settings.recognition_threshold)
    detector = MaskDetector(settings.mask_model_path)
    analyzer = RealTimeAnalyzer(recognizer, detector, attendance)

    cam = cv2.VideoCapture(args.camera_index)
    if not cam.isOpened():
        raise RuntimeError("Unable to access camera")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break

            detections = analyzer.analyze_frame(frame)
            for d in detections:
                top, right, bottom, left = d.location
                color = (0, 255, 0) if d.mask_label == "MASK" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                label = f"{d.person_name} | {d.mask_label} | {d.confidence:.2f}"
                cv2.putText(frame, label, (left, max(20, top - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            cv2.imshow("Analyzing Faces", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
