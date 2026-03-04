# Analyzing Faces

A production-style machine learning project that combines:
- Face recognition
- Automated attendance system
- Mask detection

## Architecture (High-Level)

1. `FaceRecognizer` encodes and identifies known individuals.
2. `MaskDetector` classifies mask/no-mask for each detected face.
3. `AttendanceService` records first-seen check-ins.
4. `RealTimeAnalyzer` orchestrates camera stream inference and event generation.
5. `FastAPI` service exposes REST endpoints for enrollment, attendance, and realtime analysis hooks.

## Project Structure

- `src/analyzing_faces/models`: ML models and inference wrappers
- `src/analyzing_faces/services`: attendance business logic
- `src/analyzing_faces/pipelines`: realtime orchestration
- `src/analyzing_faces/api`: API contracts and app
- `scripts`: utility scripts for run/train/bootstrap
- `tests`: unit tests

## Quick Start

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .
uvicorn analyzing_faces.api.app:app --reload
```

## Run the realtime pipeline

```bash
python scripts/run_realtime.py --camera-index 0
```

## Notes

- Add your known-face images under `data/known_faces/<person_name>/`.
- The mask detector is initialized with a baseline model wrapper; replace with a trained checkpoint for higher accuracy.
- For enterprise deployment, place event storage behind a database adapter (PostgreSQL/TimescaleDB recommended).
