import os
import json
import shutil

# === 1. Fix chunks.json ===
chunks_path = "data/chunks.json"

# If chunks.json is a folder, remove it
if os.path.isdir(chunks_path):
    print(f"Removing directory: {chunks_path}")
    shutil.rmtree(chunks_path)

# Create an empty chunks.json file
if not os.path.exists(chunks_path):
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump({}, f)
    print("✅ Created fresh empty chunks.json")

print("✅ Cleanup done. You can now safely rename files.")