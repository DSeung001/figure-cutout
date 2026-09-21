# Figure Cutout

피규어 사진에 특화된 **정밀 컷아웃(cutout) 파이프라인**을 개발하기 위한 로컬 우선 프로젝트입니다.

일반적인 배경 제거 API를 만드는 것이 목표가 아닙니다. 피규어 도메인에 맞춰 **피규어 탐지 → 정밀 segmentation → 받침대/액세서리 정책 → mask refinement → 품질 평가**를 독립적인 단계로 구성하고, 각 모델과 후처리 전략을 반복적으로 벤치마크할 수 있도록 설계합니다.

## 목표

- 로컬 GPU/CPU 자원을 최대한 활용해 모델 실험과 데이터셋 평가를 빠르게 반복
- API, 추론 파이프라인, 저장소 계층을 분리해 추후 클라우드 배포가 가능하도록 유지
- detector / segmenter / refiner / quality evaluator를 교체 가능한 컴포넌트로 설계
- 모든 실험 결과를 재현할 수 있도록 latency, throughput, 품질 지표, 환경 정보를 기록
- 실패 사례와 사용자 수정 데이터를 장기적으로 도메인 데이터셋으로 축적

자세한 제품 목표와 개발 원칙은 [agent.md](./agent.md), 아키텍처와 벤치마크 사양은 [docs/architecture.md](./docs/architecture.md)를 참고하세요.

## 권장 개발 단계

1. **Pure ML / CLI** — 단일 이미지와 데이터셋을 로컬에서 처리하고 벤치마크
2. **Local Service** — FastAPI + worker 구조로 job flow 검증
3. **Product Prototype** — 업로드, 결과 비교, mask 수정, feedback 수집
4. **Deployment** — LocalStorage/LocalQueue 구현을 S3/managed queue/GPU worker로 교체

## 프로젝트 구조

```text
figure-cutout/
├── agent.md
├── docs/
│   └── architecture.md
├── src/figure_cutout/
│   ├── api/
│   ├── ml/
│   ├── storage/
│   └── domain/
├── scripts/
│   ├── run_cutout.py
│   └── benchmark.py
├── tests/
├── data/               # gitignored
├── datasets/           # gitignored
├── models/             # gitignored
├── benchmarks/         # benchmark result JSON
├── pyproject.toml
└── .gitignore
```

## 로컬 설치

Python 3.11+ 및 `uv` 사용을 권장합니다.

```bash
uv sync
```

개발 의존성까지 설치:

```bash
uv sync --extra dev
```

ML 프레임워크는 장비/CUDA 버전에 따라 설치 방식이 달라질 수 있으므로 기본 의존성에서 분리합니다.

## 단일 이미지 실행

현재 기본 파이프라인은 실제 모델 연결 전에도 입출력 계약을 검증할 수 있는 placeholder 구현입니다.

```bash
uv run python scripts/run_cutout.py samples/figure.jpg --output data/results/figure.png
```

실제 detector / segmenter 구현은 `src/figure_cutout/ml` 인터페이스에 연결합니다.

## 벤치마크

```bash
uv run python scripts/benchmark.py \
  --input-dir datasets/figure-v1/images \
  --output benchmarks/local-baseline.json
```

벤치마크 결과에는 다음 정보를 기록합니다.

- 이미지 수
- 성공/실패 수
- p50 / p95 / 평균 latency
- throughput
- 실행 환경
- pipeline/model version
- 추후 확장 시 IoU, Boundary F-score, base/accessory policy accuracy 등 품질 지표

## 설계 원칙

**FastAPI가 모델을 소유하지 않습니다.** 추론은 별도 pipeline/worker 경계 안에서 실행합니다.

**Storage 구현에 도메인이 종속되지 않습니다.** 초기에는 local filesystem을 사용하고 추후 S3/R2로 교체합니다.

**모델 이름을 애플리케이션 로직에 하드코딩하지 않습니다.** detector, segmenter, refiner, quality evaluator는 인터페이스 뒤에 둡니다.

**벤치마크 없는 모델 교체는 하지 않습니다.** 모델/후처리 변경은 동일한 validation set에 대한 지표와 latency 기록을 남깁니다.

## 초기 완료 기준

- 로컬에서 단일 이미지 처리 가능
- 데이터셋 단위 batch benchmark 가능
- 처리 결과 및 benchmark metadata를 파일로 저장
- ML 컴포넌트를 교체할 수 있는 인터페이스 존재
- API와 ML pipeline 코드 경계 분리
- 추후 queue/storage/GPU worker를 외부 인프라로 교체할 수 있는 구조 유지
