import jwt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

SECRET = os.getenv("JWT_SECRET")
if not SECRET:
    raise RuntimeError("Missing required env var: JWT_SECRET")
if len(SECRET) < 32:
    raise RuntimeError("JWT_SECRET must be at least 32 characters long")

def generate_token(user_id):
    return jwt.encode({
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(days=7)
    }, SECRET, algorithm="HS256")

def verify_token(token):
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        return payload["user_id"]
    except:
        return None
