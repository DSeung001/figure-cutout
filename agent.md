# Agent Guide

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

## Product Scope

The pipeline must account for:

- figure body
- display base
- attached accessories
- detached accessories
- transparent or translucent effect parts
- thin structures such as hair, swords, spears, wings
- reflective or transparent boundaries
- complex shooting backgrounds

## Architecture Rules

### Local-first

Use local CPU/GPU resources for development and benchmarking.

Keep these boundaries from the start:

- API
- inference pipeline
- worker / job execution
- storage
- benchmark / evaluation
- domain models

Local implementations must be replaceable with deployed equivalents.

Examples:

```text
LocalStorage → S3 / R2
Local Queue → Redis / managed queue
Local GPU Worker → remote GPU worker
Local metadata → PostgreSQL
```

### ML Components

Keep model-specific code behind interfaces.

Replaceable components:

- Detector
- Segmenter
- MaskRefiner
- QualityEvaluator
- Storage
- Queue / JobExecutor

Do not hardcode specific model names into application logic.

### Worker

GPU inference should run in a long-lived worker process.

Goals:

- avoid repeated model loading
- avoid duplicate VRAM allocation
- keep models warm
- allow future worker scale-out

### API

The API should handle:

- validation
- job creation
- status lookup
- result delivery

Do not bind long-running GPU inference directly to the HTTP request lifecycle.

## Domain Policy

Base and accessory handling must be explicit.

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

Do not replace models or post-processing logic without a benchmark on the same validation set.

Record at minimum:

- pipeline version
- model identifiers
- dataset version
- image count
- success / failure count
- mean / p50 / p95 latency
- throughput
- peak VRAM when available
- quality metrics when ground truth exists

Preferred quality metrics:

- IoU
- Dice
- Boundary F-score
- foreground completeness
- background leakage
- base policy accuracy
- accessory policy accuracy

## Failure Data

Preserve useful failure artifacts when possible:

- original image
- detection output
- raw mask
- refined mask
- quality score
- failure reason
- model and pipeline version

Use these samples for failure analysis, hard-example mining, and future fine-tuning.

## Development Order

1. ML core: single image, batch inference, benchmark
2. Local service: API, queue, worker, job state, local storage
3. Product prototype: upload, before/after, mask correction, feedback
4. Deployment: external storage, queue, database, GPU workers

## Coding Rules

- Python 3.11+
- use type hints
- define ML input/output contracts explicitly
- separate ML code from application/service logic
- write benchmark output as machine-readable JSON
- track model and pipeline versions
- keep datasets, model weights, and runtime data out of Git
- prefer small replaceable components over framework-heavy abstractions

## Documentation Style

Documentation must be concise and implementation-oriented.

Rules:

- avoid conversational or assistant-like phrasing
- avoid long introductions, conclusions, and repeated explanations
- state goals, contracts, commands, constraints, and trade-offs directly
- prefer short sections, tables, code blocks, and bullet points
- include only information that affects implementation, operation, benchmarking, or product scope
- do not duplicate content across README, architecture docs, and this file unless required for context

README should stay minimal.
Architecture docs should describe structure and runtime decisions.
This file should define project goals and rules for future agents.

## Initial Definition of Done

- package imports successfully
- CLI can run the pipeline
- benchmark can process a dataset directory
- benchmark JSON is generated
- placeholder components support end-to-end orchestration
- real models can replace placeholders through existing interfaces
- API, worker, storage, and ML boundaries remain separate
