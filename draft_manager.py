"""
Redis-based Autosave / Draft System for Audit Copilot
"""

from config import redis_client
import json
import time
from typing import Dict, Any, Optional

DRAFT_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days

def save_draft(deck_id: str, user_id: str, data: Dict[str, Any]) -> bool:
    """
    Autosave partial audit work.
    Call this periodically from the frontend (e.g. every 30-60 seconds).
    """
    try:
        key = f"draft:{deck_id}:{user_id}"
        data["last_saved"] = time.time()
        redis_client.setex(key, DRAFT_TTL_SECONDS, json.dumps(data))
        return True
    except Exception as e:
        print(f"Failed to save draft: {e}")
        return False


def load_draft(deck_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Load the most recent draft for a deck + user"""
    try:
        key = f"draft:{deck_id}:{user_id}"
        data = redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception:
        return None


def delete_draft(deck_id: str, user_id: str) -> bool:
    """Delete draft after final submission"""
    try:
        key = f"draft:{deck_id}:{user_id}"
        redis_client.delete(key)
        return True
    except Exception:
        return False


def list_user_drafts(user_id: str) -> list:
    """List all active drafts for a user (useful for resume screen)"""
    try:
        keys = redis_client.keys(f"draft:*:{user_id}")
        drafts = []
        for key in keys:
            data = redis_client.get(key)
            if data:
                draft = json.loads(data)
                draft["deck_id"] = key.decode().split(":")[1]
                drafts.append(draft)
        return drafts
    except Exception:
        return []
