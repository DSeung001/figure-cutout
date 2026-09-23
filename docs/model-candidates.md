# Model Candidates

후보 모델 목록과 비교 절차. 진행 순서와 판단 기준은 [ml-roadmap.md](./ml-roadmap.md).

조사 기준일: 2026-09. 도입 시 라이선스·체크포인트·유지보수 상태를 다시 확인할 것.

## 선정 조건

| 조건 | 기준 |
|---|---|
| 라이선스 | 추출 결과를 서비스에 사용 → 상업 사용 가능해야 함 |
| 실행 환경 | M1 16GB (CoreML / CPU / MPS). 배치 처리이므로 장당 수십 초까지 허용 |
| 출력 | 투명 배경 RGBA, soft alpha |
| 범위 | 한 장에 피규어 1개, 피규어 본체만 추출 (좌대 제외), 경고 후 자동 반환 |

## Tier 1 — 등록 완료 (rembg ONNX)

`figure-cutout list-pipelines`로 확인. 모두 `ml/rembg_adapter.py` 하나로 동작.

| pipeline id | 모델 | 라이선스 | 크기 | 비교 목적 |
|---|---|---|---|---|
| `rembg` | U²-Net | Apache-2.0 | 176MB | 기존 기준선 |
| `rembg-isnet-general` | IS-Net (DIS) | Apache-2.0 | 179MB | 일반 DIS 기준 |
| `rembg-isnet-anime` | IS-Net anime | Apache-2.0 | 168MB | 애니 캐릭터 도메인 — 피규어 도색 스타일과 유사 |
| `rembg-birefnet-general` | BiRefNet | MIT | 928MB | **1순위 후보**. 복잡한 경계·얇은 구조 |
| `rembg-birefnet-lite` | BiRefNet lite | MIT | 214MB | 속도 대비 품질 |
| `rembg-birefnet-massive` | BiRefNet (대규모 데이터 학습) | MIT | ~930MB | 범용성 비교 |

M1 16GB 실측 (2026-09-23, 합성 512×768 3장, onnxruntime, 첫 장은 모델 로드 포함):

| pipeline id | p50 | p95 |
|---|---|---|
| `rembg-isnet-anime` | 4.2s | 10.4s |
| `rembg-birefnet-lite` | 22.4s | 29.0s |
| `rembg-birefnet-general` | 47.7s | 49.2s |

BiRefNet general 기준 100장 ≈ 80분. 배치 운영에는 충분. 속도가 문제면 CoreML provider 튜닝 또는 lite 사용.

## Tier 2 — 다음 어댑터 후보

| 모델 | 라이선스 | M1 | 이유 | 도입 방법 |
|---|---|---|---|---|
| **BEN2** | MIT | ONNX 제공 | 머리카락·경계 matting 강점 | rembg `ben_custom` 세션 + ONNX 경로, 또는 `ben2` 패키지 |
| **ToonOut** | CC BY 4.0 (가중치 조건 확인 필요) | BiRefNet 동일 | BiRefNet을 애니 캐릭터 1,228장으로 fine-tune. Pixel Accuracy 95.3→99.5% | PyTorch 가중치 → ONNX export 또는 torch 어댑터 |
| BiRefNet HR (2048) | MIT | MPS, 느림 | 고해상도 원본의 머리카락·무기 끝 | `transformers` torch 어댑터 |
| InSPyReNet | MIT | CPU/MPS | 비교군 | `transparent-background` 패키지 |

Tier 1 결과에서 `isnet-anime`이 `birefnet-general`보다 좋으면 ToonOut 우선.

## Tier 3 — 좌대 제외 (base_policy=exclude)

DIS 계열은 좌대를 피규어와 한 덩어리로 자름. Tier 1 결과에서 `base_overincluded` 빈도를 먼저 측정.

| 역할 | 모델 | 라이선스 | M1 |
|---|---|---|---|
| 좌대 위치 검출 | Grounding DINO | Apache-2.0 | 가능, 느림 |
| 좌대 위치 검출 | Florence-2 | MIT | 가능 |
| 좌대 마스크 | SAM 2.1 (small/base) | Apache-2.0 | MPS 가능 |
| 텍스트 → part 마스크 | SAM 3 | SAM License (상업 허용, 군사 용도 금지) | MPS 미지원 (2026 중반 기준), CPU만 |

방식: 전체 마스크(Tier 1/2) − 좌대 마스크(Tier 3) → `Policy` 단계.

## Tier 4 — 경계 후처리

반투명 이펙트·머리카락 실패가 많을 때만.

- trimap(마스크 erode/dilate) + ViTMatte — 라이선스 도입 시 확인
- 먼저 연결요소 필터, hole filling으로 해결되는지 측정

## 제외

| 모델 | 사유 |
|---|---|
| `bria-rmbg` (rembg 기본값) | CC BY-NC — 상업 사용 불가 |
| Ultralytics YOLO / YOLO-World | AGPL-3.0 |
| `u2net_human_seg`, `birefnet-portrait` | 사람 전용 |

## 데이터 폴더 구조

`subculture-researcher` 이미지 export를 그대로 사용. 레이아웃·제외 규칙·주의점: [export-dataset.md](./export-dataset.md)

## 결과 폴더 구조

```text
benchmarks/<pipeline-id>/<run-id>.json                      # 속도·성공/실패
data/benchmark-results/<pipeline-id>/<run-id>/000000-<key>.png   # 투명 PNG
data/debug/<run-id>/000000-<key>/                        # 마스크·품질 경고
data/compare/<stamp>/<compare-id>/
├── sheets/<key>.png           # 원본 | 모델 A | 모델 B ... (체크무늬 = 투명, 빨간 라벨 = 경고)
├── summary.json               # 모델별 run-id, latency, 경고 수
└── review.csv                 # 사람 채점표
```

`<run-id>` = `YYYYMMDDTHHMMSSZ-<hash>`, `<compare-id>` = `YYYYMMDDTHHMMSSZ` (UTC).

## 비교 절차

```bash
# 1. 최근 export 색인 (~/figure_project/exports/<stamp>)
figure-cutout init-dataset

# 2. 후보 실행 (첫 실행 시 ~/.rembg/models 에 가중치 다운로드)
figure-cutout eval \
  --pipeline rembg --pipeline rembg-isnet-anime \
  --pipeline rembg-birefnet-general --pipeline rembg-birefnet-lite

# 3. 나란히 비교
figure-cutout compare \
  --pipeline rembg --pipeline rembg-isnet-anime \
  --pipeline rembg-birefnet-general --pipeline rembg-birefnet-lite
```

`eval`에서 `--pipeline`을 생략하면 등록된 모든 파이프라인이 실행된다 (가중치 약 2.7GB).

`review.csv` 채점:

| 열 | 값 |
|---|---|
| `score_1to5` | 5 = 그대로 사용, 3 = 약간 수정 필요, 1 = 사용 불가 |
| `failure_tags` | [ml-roadmap.md](./ml-roadmap.md) Phase 4 분류, `;`로 구분 (예: `base_overincluded;thin_part_missing`) |
| `note` | 자유 기록 |
