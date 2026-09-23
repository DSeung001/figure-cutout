# AGENTS.md

## Project Goal

Figure Cutout is a figure-specific precision cutout service.

Do not treat it as a generic background-removal project.

Core pipeline:

```text
Figure Detection
→ Target Selection
→ Fine Segmentation
→ Base / Accessory Policy
→ Mask Refinement / Matting
→ Quality Evaluation
→ Confidence Fallback
```

## Required References

Before changing ML pipeline behavior, model selection, benchmarking, datasets, or training strategy, read:

- `docs/ml-roadmap.md`
- `docs/architecture.md`

Follow the ML roadmap order. Do not introduce training or a custom backbone before pretrained baselines and failure analysis justify it.

## Repository Layout

```text
figure-cutout/
├── AGENTS.md                 # repository rules and routing
├── docs/
│   ├── architecture.md       # runtime structure
│   ├── ml-roadmap.md         # model evaluation / training progression
│   ├── model-candidates.md   # candidate models, licenses, result layout
│   ├── export-dataset.md     # evaluation dataset built from image exports
│   └── image-export-format.md # export file format (owned by subculture-researcher)
├── src/figure_cutout/
│   ├── api/                  # HTTP API (validation, jobs, status, delivery)
│   ├── worker/               # long-lived GPU worker
│   ├── domain/               # domain models and policies
│   ├── ml/                   # contracts, adapters, pipeline, registry
│   ├── storage/              # storage interface + implementations
│   ├── benchmark.py
│   ├── compare.py
│   ├── dataset.py            # export / synthetic datasets, splits
│   ├── image_io.py           # shared input image decoding
│   └── cli.py
├── tests/
├── benchmarks/               # benchmark JSON (generated, not committed)
└── pyproject.toml
```

## Product Scope

Handle:

- figure body
- display base
- attached / detached accessories
- transparent or translucent effect parts
- thin structures
- reflective or transparent boundaries
- complex backgrounds

## Architecture Rules

### Local-first

Use local CPU/GPU resources for development and benchmarking.

Keep these boundaries:

- API
- inference pipeline
- worker / job execution
- storage
- benchmark / evaluation
- domain models

Deployment mappings:

```text
LocalStorage → S3 / R2
Local Queue → Redis / managed queue
Local GPU Worker → remote GPU worker
Local metadata → PostgreSQL
```

### ML Components

Keep model-specific code behind interfaces:

- Detector
- Segmenter
- MaskRefiner
- QualityEvaluator
- Storage
- Queue / JobExecutor

Do not hardcode model names into application logic.

### Worker

GPU inference belongs in a long-lived worker process.

Goals:

- avoid repeated model loading
- avoid duplicate VRAM allocation
- keep models warm
- allow worker scale-out

### API

API responsibilities:

- validation
- job creation
- status lookup
- result delivery

Do not bind long-running GPU inference directly to the HTTP request lifecycle.

The API process does not own GPU models.

### ML Pipeline

The ML pipeline must run standalone from the CLI, without the API or worker.

Register new pipelines in `ml/factory.py` and document them in `docs/model-candidates.md`.

## Domain Policy

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

Candidate ontology:

```text
figure_body
attached_accessory
detached_accessory
effect_part
display_base
background
```

## Benchmark Rules

Use the same validation split when comparing models or post-processing changes.

Record:

- pipeline version
- model identifiers
- dataset version
- image count
- success / failure count
- mean / p50 / p95 latency
- throughput
- peak VRAM when available
- quality metrics when ground truth exists
- device and runtime environment
- debug artifact path

Preferred quality metrics:

- IoU
- Dice
- Boundary F-score
- foreground completeness
- background leakage
- base policy accuracy
- accessory policy accuracy

Output locations:

```text
benchmarks/<pipeline-id>/<run-id>.json
data/benchmark-results/<pipeline-id>/<run-id>/
data/debug/<run-id>/
data/compare/<dataset>/<compare-id>/
```

Compare every model or post-processing change against benchmark results on the same split.

Preserve useful failure artifacts and corrected masks.

## Coding Rules

- Python 3.11+
- use type hints
- define ML input/output contracts explicitly
- separate ML code from application/service logic
- write benchmark output as machine-readable JSON
- track model and pipeline versions
- keep datasets, weights, and runtime data out of Git
- prefer small replaceable components

## Documentation Style

Documentation must be concise and implementation-oriented.

- no conversational or assistant-like phrasing
- no long introductions or repeated conclusions
- state goals, contracts, commands, constraints, and trade-offs directly
- prefer short sections, tables, code blocks, and bullet points
- include only implementation, operation, benchmark, or product-scope information
- avoid duplicate documentation

README stays minimal.
Architecture docs describe runtime structure.
ML roadmap owns model-evaluation and training progression.
This file owns repository rules and routing.

## Initial Definition of Done

- package imports successfully
- CLI runs the pipeline
- benchmark processes a dataset directory
- benchmark JSON is generated
- placeholders support end-to-end orchestration
- real models replace placeholders through interfaces
- API, worker, storage, and ML boundaries remain separate
