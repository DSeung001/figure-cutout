# Architecture

## Runtime

Initial local setup:

```text
CLI / API
   │
   ▼
Application Layer
   │
   ▼
FigureCutoutPipeline
   ├── Detector
   ├── Segmenter
   ├── Policy
   ├── MaskRefiner
   └── QualityEvaluator
   │
   ▼
LocalStorage
```

Service mode:

```text
Client
  │
  ▼
FastAPI
  │
  ├── Job Store
  └── Queue
        │
        ▼
    GPU Worker
        │
        ▼
FigureCutoutPipeline
        │
        ▼
     Storage
```

## Local Development

```text
Host
├── Python
├── PyTorch / CUDA
├── inference worker
└── local filesystem

Optional Docker
├── PostgreSQL
└── Redis
```

GPU runtime may stay on the host during early model iteration.

## ML Contracts

| Component | Input | Output |
|---|---|---|
| Detector | image | bbox, label, confidence |
| Segmenter | image, detection/prompt | mask, confidence |
| MaskRefiner | image, raw mask | refined mask |
| QualityEvaluator | image, detection, mask | score, fallback flag, reasons |

## Benchmark

Required metadata:

```json
{
  "pipeline_version": "dev",
  "dataset": "figure-v1",
  "device": "cuda:0",
  "image_count": 100,
  "success_count": 98,
  "failure_count": 2
}
```

Performance metrics:

- elapsed time
- mean latency
- p50 / p95 latency
- images/sec
- peak GPU memory when available

Quality metrics:

- IoU
- Dice
- Boundary F-score
- foreground completeness
- background leakage
- base policy accuracy
- accessory policy accuracy

## Dataset

```text
datasets/
└── figure-v1/
    ├── images/
    ├── masks/
    ├── metadata/
    └── splits/
        ├── train.txt
        ├── val.txt
        └── test.txt
```

Validation/test splits must remain stable across model comparisons.

## Shared Originals

`subculture-researcher` writes content-addressed originals to `$FIGURE_PROJECT_DIR/images/<prefix>/<sha256>.<ext>` and v4 export records to `exports/<stamp>/`. The default shared root is `~/figure_project`.

`export_format.py` validates v4 exports and resolves their references; dataset filtering and inference consume resolved paths. No runtime Python dependency on the producer. The shared contract is `docs/image-export-format.md`; output/cache/benchmark locations remain separate.

Directories are created only when writing files. Export sidecars preserve sample keys, manual tags and frozen splits. Originals are never garbage-collected automatically.

## Storage

Local layout:

```text
data/
├── originals/
├── results/
├── masks/
└── debug/
```

Debug artifacts:

```text
debug/<run-id>/
├── detection.json
├── raw-mask.png
├── refined-mask.png
└── metrics.json
```

Deployment mapping:

| Local | Deployment |
|---|---|
| LocalStorage | S3 / R2 |
| direct call / local queue | Redis / managed queue |
| local GPU process | GPU worker |
| file metadata | PostgreSQL |
| localhost API | containerized FastAPI |

## Scaling

Order of optimization:

1. single warm GPU worker
2. batch inference
3. CPU preprocessing parallelism
4. queue-backed execution
5. horizontal GPU worker scaling

## Initial Non-goals

- Kubernetes
- multi-region
- complex event bus
- premature microservice split
- distributed tracing

Priority: segmentation quality, latency, failure analysis, dataset growth.
