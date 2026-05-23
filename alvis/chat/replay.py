import os
import json
import random
from config import CONVERSATIONS, REPLAY_BATCH, REPLAY_STEPS, CORRECTION_STEPS
from chat.learn import learn


def load_conversations():
    if os.path.exists(CONVERSATIONS):
        with open(CONVERSATIONS, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_conversation(past, user_text, asst_text, score=1.0):
    past.append({"user": user_text, "assistant": asst_text, "score": score})
    with open(CONVERSATIONS, "w", encoding="utf-8") as f:
        json.dump(past, f, ensure_ascii=False, indent=2)


def replay(model, optimizer, past):
    if len(past) < 2:
        return
    weights = [entry.get("score", 1.0) for entry in past]
    batch   = random.choices(past, weights=weights, k=min(REPLAY_BATCH, len(past)))
    seen    = set()
    for entry in batch:
        key = entry["user"] + "\x00" + entry["assistant"]
        if key in seen:
            continue
        seen.add(key)
        steps = CORRECTION_STEPS if entry.get("score", 1.0) > 1.0 else REPLAY_STEPS
        learn(model, optimizer, entry["user"], entry["assistant"], steps=steps)
