from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from analyzing_faces.config.settings import settings
from analyzing_faces.models.face_recognizer import FaceRecognizer
from analyzing_faces.models.mask_detector import MaskDetector
from analyzing_faces.services.attendance import AttendanceService

app = FastAPI(title="Analyzing Faces API", version="0.1.0")

attendance = AttendanceService(settings.attendance_file)
face_recognizer = FaceRecognizer(settings.known_faces_dir, threshold=settings.recognition_threshold)
mask_detector = MaskDetector(settings.mask_model_path)


class EnrollmentRequest(BaseModel):
    person_name: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/attendance")
def get_attendance() -> list[dict]:
    return attendance.list_events()


@app.post("/enroll")
def enroll_person(payload: EnrollmentRequest) -> dict:
    target = settings.known_faces_dir / payload.person_name
    target.mkdir(parents=True, exist_ok=True)
    return {"message": f"Enrollment folder ready at {target}"}


@app.get("/models")
def model_info() -> dict:
    if not face_recognizer:
        raise HTTPException(status_code=500, detail="Face recognizer not initialized")

    return {
        "face_recognition_threshold": settings.recognition_threshold,
        "mask_model_path": str(settings.mask_model_path),
        "known_faces_dir": str(settings.known_faces_dir),
    }
