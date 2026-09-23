# Export 데이터셋

`subculture-researcher` 이미지 export(formatVersion 4, [image-export-format.md](./image-export-format.md))를 평가 데이터셋으로 그대로 쓴다. export 파일은 수정하지 않고 옆에 sidecar만 추가한다.

## 경로

| 항목 | 값 |
|---|---|
| 공유 폴더 | `$FIGURE_PROJECT_DIR`, 미설정 시 `~/figure_project` |
| export 위치 | `<공유 폴더>/exports/<stamp>/` (+ `<stamp>.zip`) |
| 설정 | 두 저장소의 `.env`에 같은 `FIGURE_PROJECT_DIR` |
| `--dataset` 생략 시 | `index.json`이 있는 가장 최근 `<stamp>` 폴더 |
| 데이터셋 이름 | `<stamp>` (benchmark JSON의 `dataset`, `data/compare/<stamp>/`) |

## 레이아웃

```text
~/figure_project/exports/<stamp>/
├── export.json                 # export 원본 — 수정 금지
├── index.json                  # export 원본 — 수정 금지
├── manifest.json               # init-dataset 생성: dataset 이름, 필터 설정
├── metadata/<key>.json         # tags(수동), auto_tags, item_id, role, title, URL
├── masks/<key>.png             # 정답 마스크 저장 시 생성 (Phase 6)
└── splits/val.txt              # key 목록
```

- sample id = `files[].key` (예: `FIGURE_3f2a…c9-00`). 결과·debug·compare·마스크 파일명 모두 이 값
- key → 파일 경로는 `index.json`의 `storage`/`path`로 해석한다. 폴더를 직접 훑지 않는다.
- `ok`, `skipped` 모두 평가 대상이다. SHA-256이 같은 샘플은 한 번만 포함한다.
- metadata/splits는 기록할 파일이 있을 때 생성한다. 초기화는 빈 masks/를 만들지 않는다.

## 절차

```bash
# subculture-researcher: 「이미지 다운로드」 또는 python export_images.py --ids …
figure-cutout init-dataset                      # 최근 export
figure-cutout eval --pipeline rembg-birefnet-general
figure-cutout compare --pipeline rembg-birefnet-general
```

`init-dataset` 옵션:

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--category` | `FIGURE` | 유지할 카테고리 (반복 가능). `category`(사용자 수정값) 기준, 비어 있으면 id 앞부분 |
| `--role` | `main`, `detail` | 유지할 역할 (반복 가능) |
| `--min-side` | 256 | 짧은 변 최소 px |
| `--max-aspect` | 3.0 | 긴 변 / 짧은 변 상한. detail 세로 배너 제외용 |

재실행해도 기존 `metadata/`, `splits/val.txt`, `manifest.json`은 덮어쓰지 않는다. val에 없는 새 샘플은 `unlisted_in_val`로 보고만 한다. 통과 샘플이 0개면 `val.txt`를 만들지 않는다 (필터를 고쳐 재실행).

## 제외 사유

| reason | 조건 |
|---|---|
| `not_in_library` | export 항목이 로컬 DB에 없음 (항목 id로 보고) |
| `download_error` | `status: "error"` — HTTP 실패, `not_image` 포함 |
| `category_excluded` | 카테고리가 `--category`에 없음 |
| `role_excluded` | `role`이 `--role`에 없음 |
| `duplicate` | 앞선 샘플과 `sha256` 동일 — 먼저 나온 것만 유지 |
| `missing_file` | `path`의 파일이 없음 (압축 해제 누락 등) |
| `decode_error` | 매직 바이트는 맞지만 디코딩 실패 (잘린 파일) |
| `too_small` | 짧은 변 < `--min-side` |
| `extreme_aspect` | 비율 > `--max-aspect` |

## 입력 이미지 처리

| 항목 | 처리 (`image_io.load_rgba`) |
|---|---|
| GIF / 애니메이션 WebP | 첫 프레임만 사용 |
| EXIF 회전 | 적용 후 추론 |
| CMYK / 팔레트 / 흑백 | RGBA로 변환 |
| 이미 투명한 입력 | 출력 alpha = min(입력 alpha, 마스크) — 기존 투명 영역 유지 |

## 배경 제거 관점 주의점

| 특성 | 영향 | 대응 |
|---|---|---|
| 흰 배경·누끼 상품컷 비중이 높음 | 실사 사진보다 쉬움 → 점수 과대평가 | `auto_tags`의 `plain_border`, `has_alpha`로 분리 집계 |
| detail 이미지의 텍스트·가격·배지·로고 | 전경으로 잘못 포함 | failure tag `background_leak` |
| detail 콜라주 (여러 포즈·여러 피규어) | 단일 피규어 가정 붕괴 | 메타 `tags`에 `multi_object`, 필요 시 `--role main` |
| 같은 피규어의 main / detail 여러 컷 | 샘플 간 상관 | 집계 시 `item_id`별 확인 |
| 저해상도·JPEG 압축 | 경계·얇은 부위 품질 저하 | `--min-side` 조정 |
| 좌대·박스·특전 동봉 컷 | `base_policy`/`accessory_policy` 판정 대상 | 메타 `tags`에 `display_base`, `detached_accessory` |

`auto_tags`는 자동 추정 힌트, `tags`는 사람이 입력한다 ([ml-roadmap.md](./ml-roadmap.md) Phase 2 태그).

## 운영 규칙

- 평가에 사용하는 `export.json`, `index.json`, 원본 이미지는 수정·삭제 금지. 공용 원본은 자동 삭제하지 않는다.
- export마다 새 `<stamp>` 폴더 = 새 데이터셋. 모델 비교는 같은 `<stamp>`에서만 (`--dataset`으로 고정)
- val split 생성 후 key 삭제·재생성 금지. 구성 변경 = 새 버전
- train split을 만들 때는 **`item_id` 단위로 분할**. 같은 항목의 main/detail이 train과 val에 나뉘면 누수
- formatVersion 4만 지원한다. 버전이 없거나 다른 export는 새로 내보낸다
- 두 저장소 연동 확인: `SUBCULTURE_RESEARCHER_DIR=<체크아웃 경로> pytest tests/test_cross_project_export.py` (subculture-researcher의 `.venv` 필요, 미설정 시 건너뜀)

## 저작권

- 쇼핑몰 상품 이미지. 로컬 평가 전용
- Git 커밋·외부 공유·결과물 배포 금지 (공유 폴더는 저장소 밖, `data/`는 gitignore)
- 학습(fine-tuning) 데이터로 쓰기 전 이용 권한 확인
