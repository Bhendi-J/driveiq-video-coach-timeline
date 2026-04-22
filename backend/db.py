from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
from pymongo.errors import PyMongoError

load_dotenv()
logger = logging.getLogger("driveiq.db")

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("Missing required env var: MONGO_URI")

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
db = client["DriveIQ"]

users_collection = db["users"]
sessions_collection = db["sessions"]

# Keep auth deterministic and avoid duplicate email races.
try:
    users_collection.create_index("email", unique=True)
except PyMongoError as e:
    # Do not crash the API process at import time if Mongo is temporarily down.
    logger.warning("Mongo index init skipped: %s", e)


def is_mongo_available() -> tuple[bool, str | None]:
    try:
        client.admin.command("ping")
        return True, None
    except Exception as e:
        return False, str(e)

def save_session(user_id, score, eco_score, features):
    return sessions_collection.insert_one({
        "user_id": user_id,
        "timestamp": datetime.utcnow(),
        "score": score,
        "eco_score": eco_score,
        "features": features
    })

def get_user_sessions(user_id, limit=50):
    return list(sessions_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit))
