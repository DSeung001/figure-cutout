# ML Roadmap

## Objective

Validate off-the-shelf models before training or selecting a custom backbone.

Decision order:

```text
Pretrained Baseline
→ Benchmark
→ Failure Taxonomy
→ Post-processing / Policy
→ Fine-tuning if needed
→ Backbone replacement only if justified
```

## Phase 0 — ML Basics Required

Learn only the concepts needed to operate the pipeline:

- inference: run a trained model
- checkpoint: saved model weights
- detector: locate the figure
- segmenter: produce a pixel mask
- prompt: box / point / text guidance for segmentation
- mask refinement: improve boundaries after segmentation
- fine-tuning: adapt an existing model with domain data
- backbone: core feature extractor inside a model
- ground truth: manually verified target mask
- IoU / Dice: mask overlap metrics
- latency / VRAM: runtime cost

Do not start with model training.

## Phase 1 — Build a Baseline Harness

Goal: run multiple pretrained pipelines through the same interface.

Required adapters:

```text
Detector
Segmenter
MaskRefiner
QualityEvaluator
```

Initial candidate groups:

| Baseline | Purpose |
|---|---|
| background-removal model | simple end-to-end reference |
| promptable segmentation model | boundary quality with box/point prompt |
| open-vocabulary detector + promptable segmenter | automatic figure detection + segmentation |

Candidate implementations should be selected at implementation time after checking:

- current maintenance status
- model license
- checkpoint availability
- local GPU requirements
- inference speed
- image resolution limits

Do not couple application code to a specific model family.

## Phase 2 — Create a Small Evaluation Set

Start with 50-100 representative images.

Cover:

- simple background
- cluttered background
- dark / bright background
- thin hair
- swords / spears
- wings
- display bases
- detached accessories
- transparent effect parts
- reflective / glossy surfaces
- multiple objects in frame

Initial evaluation can be visual.

Store metadata per image:

```json
{
  "id": "sample-001",
  "tags": ["sword", "display_base", "cluttered_background"],
  "difficulty": "hard"
}
```

## Phase 3 — Benchmark Pretrained Models

Run every candidate against the same dataset split.

Performance metrics:

- mean latency
- p50 latency
- p95 latency
- throughput
- peak VRAM when available
- failure count

Visual quality dimensions:

- body completeness
- thin-part preservation
- base correctness
- accessory correctness
- background leakage
- edge quality

Save:

```text
benchmarks/<model-or-pipeline>/<run-id>.json
data/debug/<run-id>/
```

Do not pick a model from a few successful screenshots.

## Phase 4 — Build a Failure Taxonomy

Classify recurring failures.

Initial categories:

```text
body_missing
body_overcut
thin_part_missing
base_missing
base_overincluded
accessory_missing
background_leak
edge_jagged
transparent_part_missing
wrong_target
multi_object_confusion
```

Track failure frequency per pipeline.

This determines what to improve next.

## Phase 5 — Improve Without Training

Before fine-tuning, test cheaper interventions:

- better detector prompt
- better bounding box selection
- multiple segmentation prompts
- mask union / intersection rules
- connected-component filtering
- hole filling
- edge refinement
- alpha matting
- explicit base/accessory policy
- fallback to a second model on low confidence

Benchmark every change against the same validation set.

## Phase 6 — Add Ground Truth

Create verified masks only after baseline failures are understood.

Start with high-value samples:

- repeated failure cases
- thin structures
- transparent effect parts
- difficult bases
- cluttered backgrounds

Recommended progression:

```text
50 verified masks
→ metric pipeline validation
→ 100-300 hard examples
→ fine-tuning experiment
```

Exact volume depends on model and failure diversity.

## Phase 7 — Fine-tune Only With Evidence

Fine-tuning is justified when:

- the same failure pattern appears across multiple pretrained models
- post-processing does not solve it reliably
- enough verified masks exist
- the expected quality gain matters to the product

Possible fine-tuning targets:

1. detector only
2. segmentation model
3. refinement / matting model

Prefer the smallest component that fixes the measured failure.

## Phase 8 — Backbone Evaluation

Do not search for a new backbone by default.

Consider backbone replacement only when:

- inference cost is unacceptable
- resolution limits block required quality
- fine-tuning plateaus
- domain failures remain structural
- deployment constraints require a smaller/faster model

Compare candidates with the same benchmark harness.

## Phase 9 — Quality Gate

A model/pipeline becomes the new baseline only when it improves target metrics without unacceptable runtime regression.

Example gate:

```text
quality target met
AND
p95 latency within budget
AND
VRAM within local/deployment budget
AND
no major regression in base/accessory handling
```

Keep the previous baseline for comparison.

## Immediate Implementation Tasks

1. keep the current placeholder pipeline
2. add a model registry / pipeline selector
3. add one end-to-end pretrained background-removal baseline
4. add one promptable segmentation baseline
5. add one detector + segmenter baseline
6. extend benchmark JSON with model/device metadata
7. add debug artifact output
8. prepare a 50-100 image evaluation set
9. record failures by category
10. decide on fine-tuning only after benchmark review
