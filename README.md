# Figure Cutout

피규어 사진에서 피규어만 투명 배경 PNG로 추출하는 파이프라인.

## 설치

Python 3.11 이상. 가상환경은 프로젝트 루트의 `.venv/` 하나를 사용한다 (Git 제외).

### A. uv 사용

```bash
uv sync --extra dev --extra ml   # .venv 생성 + 의존성 설치
uv run figure-cutout list-pipelines
```

### B. venv + pip 사용

```bash
python3 --version                  # 3.11 이상 확인
python3 -m venv .venv              # 가상환경 생성
source .venv/bin/activate          # 활성화 (프롬프트에 (.venv) 표시)
pip install -U pip
pip install -e '.[ml,dev]'         # 패키지 + 모델 + 테스트 도구
figure-cutout list-pipelines       # 설치 확인
```

| extra | 내용 |
|---|---|
| `ml` | rembg, onnxruntime (실제 모델 실행) |
| `dev` | pytest, ruff |

- 새 터미널마다 `source .venv/bin/activate` 필요. 종료는 `deactivate`
- 활성화 없이 실행: `.venv/bin/figure-cutout ...`
- 재설치: `rm -rf .venv` 후 위 절차 반복

이하 명령은 활성화된 가상환경 기준 `figure-cutout ...` 으로 표기 (uv는 `uv run figure-cutout ...`).

## 사진 한 장 처리

```bash
figure-cutout run photo.jpg --output data/results/photo.png --pipeline rembg-birefnet-general
```

출력: 투명 배경 RGBA PNG. `quality_score`, `requires_review`(경고 여부)가 함께 출력된다.

첫 실행 시 모델 가중치를 `~/.rembg/models/`에 다운로드한다.

## 모델 비교

### 1. export 경로

`subculture-researcher`의 「이미지 다운로드」 / `export_images.py`가 공유 폴더에 export(formatVersion 2)를 만든다. figure-cutout은 같은 폴더를 읽는다.

```text
~/figure_project/exports/<stamp>/     # export 1개 = 데이터셋 1개 (이름 = <stamp>)
```

- 경로 변경: 두 저장소의 `.env`에 같은 `FIGURE_PROJECT_DIR` 설정 (`.env.example` 참고)
- `--dataset`을 생략하면 가장 최근 export를 사용한다
- export 파일(`export.json`, `index.json`, `items/`)은 수정 금지

### 2. 색인

```bash
figure-cutout init-dataset
```

`metadata/<key>.json`, `splits/val.txt`가 생성된다. 기본값: `category == FIGURE`, main + detail, 짧은 변 256px 이상, 비율 3:1 이하, 중복 제거. 제외 사유별 개수가 출력된다. metadata의 `tags`에 특징을 적는다 (`sword`, `wings`, `display_base`, `transparent_effect` 등).

주의점·제외 규칙: [docs/export-dataset.md](./docs/export-dataset.md)

### 3. 모델 실행

```bash
figure-cutout eval \
  --pipeline rembg-birefnet-general \
  --pipeline rembg-isnet-anime
```

`--pipeline`을 생략하면 등록된 모든 모델을 실행한다 (가중치 약 2.7GB). 이전 export로 돌리려면 `--dataset ~/figure_project/exports/<stamp>`.

### 4. 나란히 비교

```bash
figure-cutout compare \
  --pipeline rembg-birefnet-general \
  --pipeline rembg-isnet-anime
```

```text
data/compare/<stamp>/<compare-id>/
├── sheets/<key>.png      # 원본 | 모델별 결과 (체크무늬 = 투명, 빨간 라벨 = 경고)
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
pytest            # uv: uv run pytest
```

## 문서

- [AGENTS.md](./AGENTS.md) — 저장소 규칙, 구조
- [docs/architecture.md](./docs/architecture.md) — 런타임 구조
- [docs/ml-roadmap.md](./docs/ml-roadmap.md) — 모델 평가·학습 진행 순서
- [docs/model-candidates.md](./docs/model-candidates.md) — 후보 모델, 결과 폴더
- [docs/export-dataset.md](./docs/export-dataset.md) — export 데이터셋 규칙·주의점
- [docs/image-export-format.md](./docs/image-export-format.md) — export 파일 양식 (subculture-researcher)
