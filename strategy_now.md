Strategy Now: Practical and Defensible Execution

New thesis
Build a vision-based eco-behavior analytics system from real dashcam footage, then attach model-based coaching and a trained LLM wording layer. Keep all claims bounded by what monocular video can truly support.

Core principles
1. Data truth first
- Use only measurable video-derived features (and telemetry only when actually provided).
- No fabricated speed/fuel fields.

2. Clear task separation
- Task A: behavior proxy scoring.
- Task B: issue/action coaching prediction.
- Task C: natural-language realization using a trained LLM layer.

3. Claim boundaries
- Strong claim: behavior trends and risk patterns from dashcam CV features.
- Bounded claim: score is a proxy behavior score, not true fuel efficiency.
- No claim: absolute speed/distance/fuel estimation from monocular video only.

Model stack
1. XGBoost scorer (primary)
- Input: windowed CV behavior features (+ optional telemetry if available).
- Output: per-window behavior score (0-100 proxy scale).
- Purpose: stable graph and trend backbone.

2. Coaching classifier (next)
- Input: current window + recent trend context (slope, volatility, event bursts).
- Output: top_issues, top_actions, confidence.
- Purpose: explain score drops and suggest concrete corrections.

3. Trained LLM wording layer (professor requirement)
- Input: structured coaching output from classifier.
- Output: concise, user-friendly coaching language.
- Purpose: improve communication quality while preserving factual constraints.

Execution order
1. Stabilize and freeze scoring baseline.
2. Build coaching training table with context features and weak labels.
3. Train/evaluate coaching classifier.
4. Train/adapt LLM layer for coaching text generation.
5. Integrate into coach route with fallback chain and source tagging.
6. Validate end-to-end latency, consistency, and report clarity.

Data and labeling policy
- Real footage is mandatory input.
- Weak labels are allowed only if explicitly declared.
- Human feedback labels should be added for iterative retraining.

API evolution policy
- Keep current contracts backward compatible.
- Add new fields for traceability: source, confidence, top_issues, top_actions, rationale_fields.

Success criteria
Technical
- Stable per-window score trend.
- Coaching precision@3 and macro-F1 above baseline.
- Regression tests pass before every merge/demo.

Academic
- Explicit limitation section.
- Evidence of trained LLM component.
- Reproducible artifacts and metrics.

Why this strategy is stronger
- It preserves the original product intent (moment-level scoring + coaching) while removing non-defensible physical claims.
- It gives a clear, auditable path to satisfy both engineering quality and professor expectations.
