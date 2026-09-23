# Figure Cutout

피규어 사진에서 피규어만 투명 배경 PNG로 추출하는 파이프라인.

## 설치

Python 3.11+.

```bash
uv sync --extra dev --extra ml
```

uv가 없으면:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[ml,dev]'
```

이하 명령은 `uv run figure-cutout ...` 또는 `.venv/bin/figure-cutout ...` 으로 실행.

## 사진 한 장 처리

```bash
figure-cutout run photo.jpg --output data/results/photo.png --pipeline rembg-birefnet-general
```

출력: 투명 배경 RGBA PNG. `quality_score`, `requires_review`(경고 여부)가 함께 출력된다.

첫 실행 시 모델 가중치를 `~/.rembg/models/`에 다운로드한다.

## 모델 비교

### 1. 사진 넣기

```text
datasets/figure-real-v1/images/fig-0001.jpg
datasets/figure-real-v1/images/fig-0002.jpg
...
```

- 파일명: `fig-0001.jpg` 형식, 소문자, 공백·한글 금지
- 확장자: `.jpg` / `.png` / `.webp`
- 첫 비교 이후 파일명 변경·삭제 금지. 구성을 바꾸면 `figure-real-v2`로 새로 만든다

### 2. 색인

```bash
figure-cutout init-dataset --dataset datasets/figure-real-v1
```

`metadata/<id>.json`, `splits/val.txt`가 생성된다. metadata의 `tags`에 특징을 적는다 (`sword`, `wings`, `display_base`, `transparent_effect` 등).

### 3. 모델 실행

```bash
figure-cutout eval --dataset datasets/figure-real-v1 \
  --pipeline rembg-birefnet-general \
  --pipeline rembg-isnet-anime
```

`--pipeline`을 생략하면 등록된 모든 모델을 실행한다 (가중치 약 2.7GB).

### 4. 나란히 비교

```bash
figure-cutout compare --dataset datasets/figure-real-v1 \
  --pipeline rembg-birefnet-general \
  --pipeline rembg-isnet-anime
```

```text
data/compare/figure-real-v1/<compare-id>/
├── sheets/fig-0001.png   # 원본 | 모델별 결과 (체크무늬 = 투명, 빨간 라벨 = 경고)
├── summary.json          # 모델별 속도, 성공/실패, 경고 수
└── review.csv            # 채점표: score_1to5, failure_tags, note
```

## 등록 모델

```bash
figure-cutout list-pipelines
```

| id | 모델 | M1 16GB 장당 |
|---|---|---|
| `rembg` | U²-Net | — |
| `rembg-isnet-general` | IS-Net | — |
| `rembg-isnet-anime` | IS-Net anime | ~4s |
| `rembg-birefnet-lite` | BiRefNet lite | ~22s |
| `rembg-birefnet-general` | BiRefNet | ~48s |
| `rembg-birefnet-massive` | BiRefNet massive | — |
| `placeholder` | 동작 검증용 (배경 제거 안 함) | — |

후보 모델, 라이선스, 다음 단계 모델: [docs/model-candidates.md](./docs/model-candidates.md)

## 기타 명령

| 명령 | 용도 |
|---|---|
| `bench --dataset <path> --pipeline <id>` | 모델 1개 벤치마크 |
| `prepare-dataset` | 합성 이미지 50장 생성 (동작 검증용) |

결과 위치:

```text
benchmarks/<pipeline-id>/<run-id>.json              # 속도·성공/실패
data/benchmark-results/<pipeline-id>/<run-id>/      # 결과 PNG
data/debug/<run-id>/                                # 마스크·품질 판정
```

## 테스트

```bash
.venv/bin/python -m pytest
```

## 문서

- [AGENTS.md](./AGENTS.md) — 저장소 규칙, 구조
- [docs/architecture.md](./docs/architecture.md) — 런타임 구조
- [docs/ml-roadmap.md](./docs/ml-roadmap.md) — 모델 평가·학습 진행 순서
- [docs/model-candidates.md](./docs/model-candidates.md) — 후보 모델, 폴더·파일명 규칙
