Python 3.12 Setup (Required)

Migration note: The project was migrated from Python 3.14 to Python 3.12 because PyTorch does not yet provide stable support for 3.14 in this workload, and Flan-T5 inference can crash the Flask process when calling model.generate on 3.14; Python 3.12 provides stable compatibility across PyTorch, transformers, and the current dependency stack.

Why this is required
- Python 3.14 was dropped for this project because PyTorch is not yet stable there for this workload.
- On 3.14, `model.generate()` during Flan-T5 inference can segfault and terminate the Flask process.
- Python 3.12 has stable compatibility for the current PyTorch + transformers stack.

Environment setup commands
```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Current Implementation Status (April 2026)

Objective
Build a practical dashcam-only behavior analytics system with:
- XGBoost behavior scoring for `/api/score`
- Fine-tuned Flan-T5 coaching generation for `/api/coach`

Core constraints
- No absolute speed/distance/fuel claims from monocular video-only input.
- Output is a behavior proxy score plus coaching guidance.

What is implemented

1) Scoring pipeline
- Video/CV feature extraction and schema-aligned XGBoost scoring is active.
- Runtime schema mismatch is fail-fast (`503` on `/api/score`).
- Health readiness is based on `xgb + scaler + schema_valid`.

2) Coaching dataset and model training
- Coaching dataset builder implemented:
	- `pipeline/build_coaching_dataset.py`
	- Uses dedicated coaching artifacts (`models/coaching_scaler.pkl`, `models/coaching_xgb_scorer.pkl`)
	- Writes `data/coaching_dataset.csv`
- Flan-T5 prep implemented:
	- `pipeline/prepare_flan_t5_dataset.py`
	- Writes `data/flan_t5/train.jsonl`, `val.jsonl`, `test.jsonl`
- Flan-T5 training implemented:
	- `models/train_flan_t5_coach.py`
	- Saves model to `models/flan_t5_coach`

3) Coach runtime integration
- `backend/routes/coach.py` fallback order is:
	1. local Flan-T5 generation
	2. deterministic rule-based fallback
	3. API fallback (if configured and prior paths fail)
- Response includes `source` to indicate which path produced tips.

4) Tests
- Contract tests are active for health, score, and coach routes.
- Coach tests include generation bounds and severity alignment checks.
- Current suite passes (`test.py`).

Legacy cleanup completed
- Removed old legacy files and paths:
	- `models/train_cnn_lstm.py`
	- `models/train_mobilenet.py`
	- `pipeline/lstm_prep.py`
	- `backend/routes/predict.py`
- Removed legacy runtime loading (CNN/LSTM/MobileNet) from `backend/model_loader.py`.
- Removed `/api/predict` registration from `backend/app.py`.
- Removed TensorFlow/MobileNet path from active CV pipeline.

Current active API surface
- `GET /api/health`
- `POST /api/score`
- `POST /api/coach`

Known runtime note
- Some environments still show backend process kills (`exit 137`) under memory pressure.
- This is environment/runtime pressure, not a test failure in the codebase.

Primary implementation files
- Scoring/runtime:
	- `backend/model_loader.py`
	- `backend/routes/score.py`
	- `backend/routes/health.py`
	- `cv/cv_pipeline.py`
- Coaching:
	- `backend/routes/coach.py`
	- `pipeline/build_coaching_dataset.py`
	- `pipeline/prepare_flan_t5_dataset.py`
	- `models/train_flan_t5_coach.py`
- Validation:
	- `test.py`