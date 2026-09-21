# Figure Cutout Architecture

## 1. Design Goal

초기에는 로컬 머신의 GPU/CPU를 적극 활용한다. 다만 코드 구조는 추후 API 서버, queue, GPU worker, object storage로 확장 가능해야 한다.

핵심은 **로컬 구현과 배포 구현이 동일한 도메인 인터페이스를 사용하도록 하는 것**이다.

## 2. Logical Architecture

```text
CLI / API
   │
   ▼
Application Service
   │
   ▼
FigureCutoutPipeline
   ├── Detector
   ├── Segmenter
   ├── Policy
   ├── Refiner
   └── QualityEvaluator
   │
   ▼
Storage
```

서비스 모드에서는 다음으로 확장한다.

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

## 3. Local-first Runtime

초기 권장 실행:

```text
Host OS
├── Python application
├── PyTorch / CUDA
├── GPU inference worker
└── local filesystem

Optional Docker
├── PostgreSQL
└── Redis
```

ML iteration 속도를 위해 GPU runtime을 반드시 Docker 안에 넣을 필요는 없다.

## 4. ML Component Contracts

### Detector

입력:

- image

출력:

- bbox
- confidence
- label

### Segmenter

입력:

- image
- optional bbox/prompt

출력:

- mask
- confidence

### MaskRefiner

입력:

- image
- raw mask

출력:

- refined mask

### QualityEvaluator

입력:

- image
- detection
- mask

출력:

- normalized score
- review/fallback 여부
- reason list

## 5. Benchmark Specification

벤치마크는 모델 품질과 실행 성능을 함께 측정한다.

### Required metadata

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

### Performance metrics

- total elapsed time
- mean latency
- p50 latency
- p95 latency
- images/sec
- optional GPU peak memory

### Quality metrics

Ground truth가 있을 경우:

- IoU
- Dice
- Boundary F-score

도메인 평가를 추가할 경우:

- body completeness
- base inclusion accuracy
- accessory inclusion accuracy
- background leakage rate

## 6. Benchmark Dataset Rules

validation/test dataset은 모델 실험 중 임의로 섞지 않는다.

권장 구조:

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

실험 로그에는 반드시 dataset/version 또는 commit/hash를 남긴다.

## 7. Storage Layout

초기 LocalStorage:

```text
data/
├── originals/
├── results/
├── masks/
└── debug/
```

debug artifact 예시:

```text
debug/<run-id>/
├── detection.json
├── raw-mask.png
├── refined-mask.png
└── metrics.json
```

추후 object storage에서도 동일한 logical key 구조를 유지한다.

## 8. Deployment Mapping

| Local | Deployment |
|---|---|
| LocalStorage | S3 / R2 |
| direct function call | queue |
| local GPU process | GPU worker instance |
| JSON/file metadata | PostgreSQL |
| localhost API | containerized FastAPI |

## 9. Scaling Strategy

초기에는 vertical scaling을 우선한다.

1. 단일 GPU worker
2. model warm-load
3. batch 처리 최적화
4. CPU preprocessing 병렬화
5. 필요 시 worker 수평 확장

worker 수평 확장 시 job queue를 공유하고 결과 저장소를 외부화한다.

## 10. Non-goals for Initial Stage

초기에는 다음을 우선하지 않는다.

- Kubernetes
- multi-region
- autoscaling controller
- complex event bus
- microservice 분할 자체
- premature distributed tracing

먼저 ML 품질, latency, 실패 패턴, 데이터셋 축적 루프를 검증한다.
