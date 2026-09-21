# Figure Cutout

피규어 전용 정밀 컷아웃 파이프라인.

일반 배경 제거가 아니라 다음 단계를 분리해 품질과 교체 가능성을 확보한다.

```text
Figure Detection
→ Fine Segmentation
→ Base / Accessory Policy
→ Mask Refinement
→ Quality Evaluation
→ Confidence Fallback
```

## 목표

- 로컬 GPU/CPU 기반 빠른 실험
- detector / segmenter / refiner 교체 가능 구조
- 동일 데이터셋 기준 benchmark
- 실패 사례와 수정 mask 축적
- 추후 API / worker / object storage 구조로 확장 가능

## 구조

```text
figure-cutout/
├── AGENTS.md
├── docs/architecture.md
├── src/figure_cutout/
│   ├── api/
│   ├── worker/
│   ├── domain/
│   ├── ml/
│   └── storage/
├── scripts/
│   ├── run_cutout.py
│   └── benchmark.py
├── tests/
├── benchmarks/
└── pyproject.toml
```

## 로컬 실행

```bash
uv sync --extra dev
```

단일 이미지:

```bash
uv run python scripts/run_cutout.py   samples/figure.jpg   --output data/results/figure.png
```

벤치마크:

```bash
uv run python scripts/benchmark.py   --input-dir datasets/figure-v1/images   --output benchmarks/local-baseline.json
```

현재 placeholder pipeline은 orchestration 검증용이다. 실제 모델은 `src/figure_cutout/ml` 인터페이스에 연결한다.

## Benchmark

기본 기록 항목:

- pipeline / model version
- image count
- success / failure
- mean / p50 / p95 latency
- throughput
- runtime environment

추가 품질 지표:

- IoU
- Dice
- Boundary F-score
- base policy accuracy
- accessory policy accuracy

## 설계 원칙

- FastAPI 프로세스가 GPU 모델을 직접 소유하지 않는다.
- ML pipeline은 CLI에서도 독립 실행 가능해야 한다.
- storage 구현은 인터페이스 뒤에 둔다.
- 모델명은 application logic에 하드코딩하지 않는다.
- 모델/후처리 변경은 benchmark 결과와 함께 비교한다.

상세 기준은 [AGENTS.md](./AGENTS.md), ML 진행 순서는 [docs/ml-roadmap.md](./docs/ml-roadmap.md), 구조는 [docs/architecture.md](./docs/architecture.md) 참고.
