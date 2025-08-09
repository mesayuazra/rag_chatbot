import os
import json

HISTORY_FILE = "data/chat_history.json"
os.makedirs("data", exist_ok=True)

def save_to_history(question, answer):
  entry = {"question": question, "answer": answer}
  history = []

  if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
      try:
        history = json.load(f)
      except json.JSONDecodeError:
        history = []

  history.append(entry)

  with open(HISTORY_FILE, "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
  if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
      try:
        return json.load(f)
      except json.JSONDecodeError:
        return []
  return []