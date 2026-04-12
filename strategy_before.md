Strategy Before: Original Direction and Why It Became Risky

Original intent
- Use dashcam footage to estimate eco score moments across the full clip.
- Consider sequence models (especially LSTM) for keyframe-series input and timeline JSON output.
- Use generated score trend to drive coaching.

What we expected initially
- Whole-video/keyframe processing would produce a generalized eco score graph.
- LSTM would simplify sequence handling for trend outputs.
- Coaching could be attached after score generation.

Assumptions that broke
1. Physical variable inference assumption
- We implicitly treated monocular video as if it could provide absolute speed, distance, and fuel behavior.
- In practice, dashcam-only video does not provide reliable absolute speed/fuel ground truth.

2. Label realism assumption
- Score labels were treated as if they represented true eco/fuel outcomes.
- Without telemetry or human labels, these are proxy labels, not physical truth labels.

3. Objective mixing assumption
- We mixed two tasks: numerical scoring and natural-language coaching.
- This blurred evaluation and made model claims hard to defend academically.

Observed risks
- Scientific validity risk: over-claiming what vision-only data can infer.
- Label risk: weak labels presented as true targets.
- Presentation risk: unclear boundaries during viva/demo.
- Engineering risk: investing in complex sequence models without stronger supervision.

What remained useful from the old plan
- Temporal windowing and trend graph requirement.
- Frame-chain processing and session continuity.
- CV feature extraction pipeline (YOLO + optical flow).
- User requirement for actionable coaching tied to trend changes.

Summary of why this plan was not enough
- The old direction had a good UX goal but weak measurement assumptions.
- It needed a stricter data-truth policy and a clearer separation between behavior proxy scoring and physical eco metrics.
