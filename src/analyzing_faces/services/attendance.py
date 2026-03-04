from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class AttendanceEvent:
    person_name: str
    timestamp: datetime
    confidence: float
    mask_label: str


class AttendanceService:
    def __init__(self, attendance_file: Path) -> None:
        self.attendance_file = attendance_file
        self.attendance_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.attendance_file.exists():
            pd.DataFrame(columns=["person_name", "timestamp", "confidence", "mask_label"]).to_csv(
                self.attendance_file,
                index=False,
            )

    def record_once_per_day(self, event: AttendanceEvent) -> bool:
        df = pd.read_csv(self.attendance_file)
        event_day = event.timestamp.date().isoformat()

        already_present = (
            (df["person_name"] == event.person_name)
            & (pd.to_datetime(df["timestamp"]).dt.date.astype(str) == event_day)
        ).any()
        if already_present:
            return False

        row = {
            "person_name": event.person_name,
            "timestamp": event.timestamp.isoformat(),
            "confidence": round(event.confidence, 4),
            "mask_label": event.mask_label,
        }
        pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(self.attendance_file, index=False)
        return True

    def list_events(self) -> list[dict]:
        return pd.read_csv(self.attendance_file).to_dict(orient="records")
