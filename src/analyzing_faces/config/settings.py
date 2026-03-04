from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[3]
    known_faces_dir: Path = project_root / "data" / "known_faces"
    attendance_file: Path = project_root / "data" / "attendance.csv"
    mask_model_path: Path = project_root / "artifacts" / "mask_detector.keras"
    recognition_threshold: float = 0.52


settings = Settings()
