"""
backend/app.py
DriveIQ Flask API entry point.

Start with:
    cd /Users/jatinankushnimje/Documents/Coding/driveiq_practice
    source .venv/bin/activate
    python backend/app.py

Or with gunicorn (production):
    gunicorn -w 2 -b 0.0.0.0:5000 backend.app:app
"""

import sys
import os
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask
from flask_cors import CORS

from backend.model_loader import load_models
from backend.routes.health  import health_bp
from backend.routes.score   import score_bp
from backend.routes.coach   import coach_bp
from backend.routes.coach   import _load_flan_model


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, origins=["http://localhost:5173", "http://localhost:3000"])

    # Load models once at startup and store in app config
    app.config["MODELS"] = load_models()

    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(score_bp)
    app.register_blueprint(coach_bp)

    # Load coach model synchronously at startup so first request has no race.
    _load_flan_model()

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print(f"\n🚗 DriveIQ API running on http://localhost:{port}")
    print("   Endpoints:")
    print("     GET  /api/health")
    print("     POST /api/score")
    print("     POST /api/coach")
    app.run(host="0.0.0.0", port=port, debug=debug)
