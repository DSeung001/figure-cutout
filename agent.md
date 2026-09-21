# Agent Guide — Figure Cutout

## Service Mission

Figure Cutout은 일반적인 "배경 제거 서비스"가 아니다.

이 프로젝트의 목표는 **피규어 전용 정밀 컷아웃 서비스**를 구축하는 것이다. 모델 선택, 데이터셋 설계, API 구조, 벤치마크, 후처리 정책은 모두 이 목표에 맞춰야 한다.

핵심 처리 흐름은 다음과 같다.

```text
Input Image
  ↓
Figure Detection
  ↓
Target Selection
  ↓
Fine Segmentation
  ↓
Base / Accessory Policy
  ↓
Mask Refinement / Matting
  ↓
Quality Evaluation
  ↓
Confidence Fallback
  ↓
Cutout Result
```

## Product Principles

### 1. General background removal is not the objective

단일 generic segmentation 모델을 호출하는 구현으로 끝내지 않는다.

필수적으로 고려해야 할 도메인 요소:

- 피규어 본체
- 받침대
- 결합 액세서리
- 분리 액세서리
- 투명/반투명 이펙트 파츠
- 얇은 구조물(검, 창, 머리카락, 날개 등)
- 복잡한 촬영 배경
- 유광/투명 재질의 경계

### 2. Local-first, deployable-by-design

초기 개발과 벤치마크는 로컬 GPU/CPU를 적극 사용한다.

하지만 아래 경계는 처음부터 유지한다.

- API layer
- inference pipeline
- worker/job execution
- storage abstraction
- benchmark/evaluation
- domain model

로컬 구현은 최종 인프라의 "축소판"이어야 한다. 추후 cloud storage, managed queue, GPU worker로 교체할 수 있어야 한다.

### 3. Benchmark before replacement

모델이나 후처리를 교체할 때는 반드시 동일한 validation set에서 비교한다.

최소 기록 항목:

- pipeline version
- model identifiers
- dataset version
- image count
- success/failure count
- mean latency
- p50/p95 latency
- throughput
- peak VRAM (측정 가능한 경우)
- quality metrics

향후 품질 지표:

- Mask IoU
- Dice/F1
- Boundary F-score
- foreground completeness
- background leakage
- base policy accuracy
- accessory policy accuracy

### 4. Interfaces over model names

비즈니스 로직에 특정 모델명을 직접 박지 않는다.

교체 가능한 컴포넌트:

- Detector
- Segmenter
- MaskRefiner
- QualityEvaluator
- Storage
- Queue / JobExecutor

### 5. Failure is data

실패 사례는 삭제하지 않는다.

가능한 경우 다음을 보존한다.

- original image
- detection output
- raw mask
- refined mask
- quality score
- failure reason
- model/pipeline version

이 데이터는 hard-example mining과 fine-tuning의 기반이 된다.

### 6. Base and accessory policy must be explicit

받침대/액세서리 포함 여부를 암묵적으로 처리하지 않는다.

예상 정책:

```text
base_policy:
- include
- exclude
- auto

accessory_policy:
- include_all
- include_attached
- exclude
```

향후 ontology 후보:

```text
figure_body
attached_accessory
detached_accessory
effect_part
display_base
background
```

## Architecture Rules

### API

- API 프로세스가 GPU 모델을 직접 소유하지 않도록 한다.
- API는 request validation, job 생성, 상태 조회를 담당한다.
- 장시간 inference를 HTTP request lifecycle에 묶지 않는다.

### ML Pipeline

ML pipeline은 API 없이 독립적으로 실행 가능해야 한다.

최소 실행 형태:

```bash
python scripts/run_cutout.py input.jpg --output output.png
```

데이터셋 단위 실행:

```bash
python scripts/benchmark.py --input-dir datasets/<name>/images
```

### Storage

스토리지 인터페이스를 통해 파일 접근을 추상화한다.

초기 구현:

```text
LocalStorage
```

배포 시 구현 가능:

```text
S3Storage
R2Storage
```

도메인/ML 코드는 storage provider를 몰라야 한다.

### Worker

GPU 사용은 기본적으로 단일 long-lived worker 프로세스가 담당한다.

이유:

- 모델 반복 로딩 방지
- VRAM 중복 점유 방지
- warm model 유지
- 추후 GPU worker scale-out 용이

## Development Order

### Phase 1 — ML Core

목표:

- single image inference
- batch inference
- benchmark
- debug artifact 저장

### Phase 2 — Local Service

목표:

- FastAPI
- local queue or Redis
- inference worker
- job state
- LocalStorage

### Phase 3 — Product Prototype

목표:

- upload
- before/after
- mask correction
- feedback
- annotation collection

### Phase 4 — Deployment

교체 대상:

```text
LocalStorage → S3/R2
Local Queue → Redis/managed queue
Local Worker → remote GPU worker
Local DB → managed PostgreSQL
```

## Coding Guidelines

- Python 3.11+ 기준
- type hint 적극 사용
- 모델 입출력 타입을 dataclass/protocol로 명확히 정의
- ML 구현과 application/service 로직 분리
- benchmark output은 machine-readable JSON으로 남김
- 모델 버전과 pipeline 버전 추적 가능하게 유지
- data/, datasets/, models/는 기본적으로 Git에 커밋하지 않음
- 작은 단위로 교체 가능하고 테스트 가능한 코드를 선호

## Definition of Done for Initial Skeleton

초기 골격은 다음을 만족해야 한다.

1. package import가 정상 동작한다.
2. CLI에서 pipeline을 호출할 수 있다.
3. dataset directory를 대상으로 benchmark를 실행할 수 있다.
4. benchmark JSON을 생성할 수 있다.
5. 실제 ML 모델이 없어도 placeholder/mock pipeline으로 end-to-end 흐름을 검증할 수 있다.
6. 실제 모델은 동일 인터페이스를 구현하여 교체할 수 있다.
7. API/worker/storage 확장을 위한 디렉터리 경계가 존재한다.
