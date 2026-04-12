"""
backend/routes/coach.py
POST /api/coach
Body: { "score": float, "features": dict, "predicted_fuel_rate": float, "history_summary": str }

Returns: { "message": str, "tips": [str, str, str], "severity": str }
"""

from __future__ import annotations

import os
import time
import threading
from pathlib import Path
from flask import Blueprint, request, jsonify

coach_bp = Blueprint("coach", __name__)

# Cache to avoid repeated identical API calls while preserving personalization.
_tip_cache: dict = {}
COACH_CACHE_TTL_SEC = 120
COACH_CACHE_MAX = 256

_flan_cache: dict = {
    "model": None,
    "tokenizer": None,
    "load_error": None,
    "status": "idle",  # idle | loading | ready | failed | disabled
}
_flan_load_lock = threading.Lock()
_flan_generate_lock = threading.Lock()

FLAN_GENERATE_TIMEOUT_SEC = float(os.environ.get("DRIVEIQ_FLAN_GENERATE_TIMEOUT_SEC", "30"))


def _make_prompt(score: float, features: dict, predicted_fuel_rate: float, history_summary: str = "") -> str:
    top_issues = []
    if features.get("braking_flag"):
        top_issues.append("hard braking detected")
    if features.get("lane_change_flag"):
        top_issues.append("frequent lane changes")
    if float(features.get("speed", 0)) > 110:
        top_issues.append("speed too high (>110 km/h)")
    if features.get("proximity_score", 0) > 0.15:
        top_issues.append("driving too close to the vehicle ahead")
    if not top_issues:
        top_issues.append("general driving pattern")

    issues_str = ", ".join(top_issues)
    return (
        f"Driving score: {score:.0f}/100\n"
        f"Top issues: {issues_str}\n"
        f"Predicted next fuel rate: {predicted_fuel_rate:.2f} L/100km\n\n"
        f"History summary: {history_summary or 'N/A'}\n\n"
        "Give exactly 3 short, actionable eco-driving tips to improve the score. "
        "Return ONLY the 3 tips as a numbered list (1. ... 2. ... 3. ...). "
        "Each tip should be one sentence."
    )


def _parse_tips(text: str) -> list[str]:
    """Extract numbered tips from Claude response."""
    lines = text.strip().splitlines()
    tips = []
    for line in lines:
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            tip = line.split(".", 1)[-1].strip()
            if tip:
                tips.append(tip)
    # Ensure exactly 3 tips
    while len(tips) < 3:
        tips.append("Maintain a steady, consistent speed for better efficiency.")
    return tips[:3]


def _load_flan_model():
    """Lazy-load locally fine-tuned Flan-T5 model/tokenizer if available."""
    model_dir = Path(__file__).resolve().parent.parent.parent / "models" / "flan_t5_coach"

    if os.environ.get("DRIVEIQ_DISABLE_FLAN_COACH", "0") == "1":
        _flan_cache["status"] = "disabled"
        _flan_cache["load_error"] = "disabled by DRIVEIQ_DISABLE_FLAN_COACH"
        return None, None

    # Fast path when already loaded.
    if _flan_cache.get("status") == "ready":
        if _flan_cache.get("model") is not None and _flan_cache.get("tokenizer") is not None:
            return _flan_cache.get("model"), _flan_cache.get("tokenizer")
        # Defensive self-heal in case of inconsistent cache state.
        _flan_cache["status"] = "failed"
        _flan_cache["load_error"] = "inconsistent ready cache state"

    # Retry only for recoverable prior failures.
    if _flan_cache.get("status") == "failed":
        load_error = str(_flan_cache.get("load_error", ""))
        if load_error.startswith("missing model dir") and model_dir.exists():
            _flan_cache["status"] = "idle"
            _flan_cache["load_error"] = None
        elif load_error == "interrupted/partial flan load":
            _flan_cache["status"] = "idle"
            _flan_cache["load_error"] = None
        else:
            return None, None

    # Another request is currently loading; do not block request threads.
    if _flan_cache.get("status") == "loading":
        return None, None

    with _flan_load_lock:
        # Re-check after acquiring lock (another thread may have loaded it).
        if _flan_cache.get("status") == "ready":
            return _flan_cache.get("model"), _flan_cache.get("tokenizer")

        if not model_dir.exists():
            _flan_cache["status"] = "failed"
            _flan_cache["load_error"] = f"missing model dir: {model_dir}"
            _flan_cache["model"] = None
            _flan_cache["tokenizer"] = None
            print(f"[flan_loader] ❌ model dir missing: {model_dir}")
            return None, None

        _flan_cache["status"] = "loading"
        _flan_cache["load_error"] = None
        _flan_cache["model"] = None
        _flan_cache["tokenizer"] = None

        try:
            print(f"[flan_loader] Loading from {model_dir} ...")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            print("[flan_loader]   Loading tokenizer ...")
            tok = AutoTokenizer.from_pretrained(str(model_dir))
            print("[flan_loader]   ✅ Tokenizer loaded")
            print("[flan_loader]   Loading model weights ...")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(model_dir),
                torch_dtype="auto",
            )
            print("[flan_loader]   ✅ Model loaded")
            _flan_cache["model"] = model
            _flan_cache["tokenizer"] = tok
            _flan_cache["status"] = "ready"
            print("[flan_loader] ✅ Flan-T5 coach ready")
        except Exception as e:
            _flan_cache["load_error"] = str(e)
            _flan_cache["model"] = None
            _flan_cache["tokenizer"] = None
            _flan_cache["status"] = "failed"
            print(f"[flan_loader] ❌ Load failed: {e}")

    return _flan_cache.get("model"), _flan_cache.get("tokenizer")


def warmup_flan_async() -> None:
    """Start Flan warmup in a background thread if it is not loaded yet."""
    status = _flan_cache.get("status")
    if status in {"loading", "ready", "disabled"}:
        return

    def _warmup():
        _load_flan_model()

    t = threading.Thread(target=_warmup, name="driveiq-flan-warmup", daemon=True)
    t.start()


def get_coach_status() -> dict:
    """Expose current coach model readiness for health endpoint."""
    return {
        "status": _flan_cache.get("status", "idle"),
        "ready": _flan_cache.get("status") == "ready",
        "disabled": _flan_cache.get("status") == "disabled",
        "load_error": _flan_cache.get("load_error"),
    }


def _primary_and_secondary_issue(features: dict) -> tuple[str, str]:
    issue_scores = {
        "proximity_score": float(features.get("proximity_score", 0.0)),
        "braking_flag": float(features.get("braking_flag", 0.0)) * 1.0,
        "lane_change_flag": float(features.get("lane_change_flag", 0.0)) * 1.0,
        "mean_flow": abs(float(features.get("mean_flow", 0.0))),
        "flow_variance": abs(float(features.get("flow_variance", 0.0))),
    }
    ranked = sorted(issue_scores.items(), key=lambda kv: kv[1], reverse=True)
    top1 = ranked[0][0] if ranked else "general_pattern"
    top2 = ranked[1][0] if len(ranked) > 1 else top1
    return top1, top2


def _build_flan_input(score: float, features: dict, history_summary: str = "") -> str:
    severity = _severity_from_score(score)
    top1, top2 = _primary_and_secondary_issue(features)

    return (
        "Task: Generate one concise and actionable eco-driving coaching sentence.\n"
        f"predicted_score: {float(score):.2f}\n"
        f"top_issue: {top1}\n"
        f"top_2_issue: {top2}\n"
        f"severity: {severity}\n"
        f"rpm_variation: {float(features.get('rpm_variation', 0.0)):.4f}\n"
        f"harsh_braking_count: {float(features.get('harsh_braking_count', 0.0)):.4f}\n"
        f"idling_time: {float(features.get('idling_time', 0.0)):.4f}\n"
        f"fuel_consumption: {float(features.get('fuel_consumption', 0.0)):.4f}\n"
        f"acceleration_smoothness: {float(features.get('acceleration_smoothness', 0.0)):.4f}\n"
        f"history_summary: {history_summary or 'N/A'}\n"
        "Response:"
    )


def _generate_flan_tip(score: float, features: dict, history_summary: str = "") -> tuple[str | None, str]:
    """Returns (tip_text_or_None, debug_reason)."""
    warmup_flan_async()

    status = _flan_cache.get("status")
    if status in {"idle", "loading"}:
        return None, f"flan_not_ready: {status}"

    model, tok = _load_flan_model()
    if model is None or tok is None:
        reason = _flan_cache.get("load_error") or "model/tokenizer is None (unknown)"
        return None, f"flan_load_failed: {reason}"

    prompt = _build_flan_input(score, features, history_summary)
    if not _flan_generate_lock.acquire(timeout=0.05):
        return None, "flan_generate_busy"

    try:
        result: dict = {}
        error: dict = {}

        def _run_generation():
            try:
                enc = tok(prompt, return_tensors="pt", truncation=True, max_length=192)
                out = model.generate(**enc, max_new_tokens=48, num_beams=1)
                text = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
                result["text"] = text
            except Exception as e:
                error["err"] = str(e)

        t = threading.Thread(target=_run_generation, name="driveiq-flan-generate", daemon=True)
        t.start()
        t.join(FLAN_GENERATE_TIMEOUT_SEC)

        if t.is_alive():
            return None, f"flan_generate_timeout>{FLAN_GENERATE_TIMEOUT_SEC}s"

        if "err" in error:
            return None, f"flan_generate_exception: {error['err']}"

        text = str(result.get("text", "")).strip()
        if text:
            return text, "flan_ok"
        return None, "flan_generated_empty_string"
    finally:
        _flan_generate_lock.release()


def _cache_key(score: float, features: dict, pred_fuel: float, history_summary: str, session_id: str = "") -> str:
    """Build a behavior-aware cache key to avoid stale/non-personalized coaching."""
    score_bucket = int(round(score / 5.0) * 5)
    fuel_bucket = int(round(pred_fuel * 2.0) / 2.0 * 10)

    braking = int(bool(features.get("braking_flag", 0)))
    lane = int(bool(features.get("lane_change_flag", 0)))
    ped = int(bool(features.get("pedestrian_flag", 0)))
    proximity_bucket = int(min(9, max(0, float(features.get("proximity_score", 0.0)) * 10)))
    road = int(features.get("road_type_id", 0)) if str(features.get("road_type_id", "")).replace(".", "", 1).isdigit() else 0
    weather = int(features.get("weather_id", 0)) if str(features.get("weather_id", "")).replace(".", "", 1).isdigit() else 0

    hs = (history_summary or "").strip().lower()
    hs_token = hs[:40]

    sid = (session_id or "").strip()
    return (
        f"s{score_bucket}|f{fuel_bucket}|b{braking}|l{lane}|p{ped}|"
        f"prox{proximity_bucket}|r{road}|w{weather}|h{hs_token}|sid{sid}"
    )


def _cache_get(key: str):
    entry = _tip_cache.get(key)
    if not entry:
        return None

    if time.time() - float(entry.get("ts", 0.0)) > COACH_CACHE_TTL_SEC:
        _tip_cache.pop(key, None)
        return None

    payload = dict(entry.get("payload", {}))
    payload["cached"] = True
    return payload


def _cache_set(key: str, payload: dict):
    if len(_tip_cache) >= COACH_CACHE_MAX and key not in _tip_cache:
        oldest_key = min(_tip_cache, key=lambda k: _tip_cache[k].get("ts", 0.0))
        _tip_cache.pop(oldest_key, None)

    _tip_cache[key] = {"ts": time.time(), "payload": payload}


@coach_bp.route("/api/coach", methods=["POST"])
def coach():
    data      = request.get_json(silent=True) or {}
    score     = float(data.get("score", 50))
    features  = data.get("features", {})
    pred_fuel = float(data.get("predicted_fuel_rate", 0))
    history_summary = str(data.get("history_summary", ""))
    session_id = str(data.get("session_id", "") or request.headers.get("X-Session-Id", ""))

    cache_key = _cache_key(score, features, pred_fuel, history_summary, session_id)
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(cached)

    # 1) Primary path: local fine-tuned Flan-T5 coaching model.
    flan_tip, flan_debug = _generate_flan_tip(score, features, history_summary)
    if flan_tip:
        tips = [flan_tip]
        # Fill remaining slots with deterministic rules for robustness.
        for t in _fallback_tips(score, features):
            if t not in tips:
                tips.append(t)
            if len(tips) >= 3:
                break
        payload = _build_response_payload(score, tips[:3])
        payload["source"] = "flan_t5"
        payload["debug_flan_reason"] = flan_debug
        _cache_set(cache_key, payload)
        return jsonify(payload)

    # 2) Secondary fallback: deterministic rules.
    try:
        tips = _fallback_tips(score, features)
        payload = _build_response_payload(score, tips)
        payload["source"] = "rules"
        payload["fallback"] = True
        payload["debug_flan_reason"] = flan_debug
        _cache_set(cache_key, payload)
        return jsonify(payload)

    except Exception:
        # Final fallback: deterministic defaults only.
        tips = [
            "Maintain a steady, consistent speed for better efficiency.",
            "Avoid abrupt braking by anticipating traffic earlier.",
            "Apply throttle smoothly instead of in sharp bursts.",
        ]

    payload = _build_response_payload(score, tips)
    payload["source"] = "rules"
    payload["fallback"] = True
    _cache_set(cache_key, payload)
    return jsonify(payload)


def _fallback_tips(score: float, features: dict) -> list[str]:
    """Rule-based fallback tips when Claude API is unavailable."""
    tips = []
    if features.get("braking_flag"):
        tips.append("Anticipate traffic earlier and ease off the accelerator gradually.")
    if features.get("lane_change_flag"):
        tips.append("Minimize unnecessary lane changes — they increase fuel burn by 5-10%.")
    if float(features.get("speed", 0)) > 110:
        tips.append("Reduce speed to below 100 km/h — fuel consumption rises sharply above 110.")
    if features.get("proximity_score", 0) > 0.15:
        tips.append("Increase following distance to allow smoother acceleration/deceleration.")

    defaults = [
        "Maintain a steady speed using cruise control on highways.",
        "Shift to a higher gear early to keep RPM low.",
        "Check tyre pressure weekly — underinflation increases fuel use by up to 3%.",
    ]
    for d in defaults:
        if len(tips) >= 3:
            break
        if d not in tips:
            tips.append(d)

    return tips[:3]


def _severity_from_score(score: float) -> str:
    if score >= 75:
        return "green"
    if score >= 50:
        return "yellow"
    return "red"


def _build_response_payload(score: float, tips: list[str]) -> dict:
    severity = _severity_from_score(score)
    message = f"Current eco-driving status is {severity}; focus on the tips below to improve efficiency."
    return {
        "message": message,
        "tips": tips,
        "severity": severity,
    }
