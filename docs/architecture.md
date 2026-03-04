# System Design Notes

## Core Components

- Inference Edge Layer: camera streams and lightweight preprocessing.
- Recognition Layer: known-face embedding lookup with distance thresholding.
- Compliance Layer: mask classifier on cropped face regions.
- Event Layer: attendance event dedupe and persistence.
- API Layer: enrollment, attendance retrieval, health checks.

## Scalability Path

- Swap local CSV storage with PostgreSQL + event table partitioning.
- Add Kafka (or Redis Streams) for frame event buffering.
- Split recognition and mask services for horizontal scaling.
- Introduce model registry for checkpoint promotion and rollback.
- Add observability: Prometheus metrics + OpenTelemetry tracing.
