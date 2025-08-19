import os
import shutil
import json

def reset_uploads():
    folder = "uploads"
    if os.path.exists(folder):
        shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        print(f"Reset folder: {folder} ✅")
    else:
        print(f"Folder {folder} tidak ditemukan ❌")

def reset_chunks():
    chunks_file = "chunks.json"
    if os.path.exists(chunks_file):
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print(f"Reset {chunks_file} ✅")
    else:
        print(f"{chunks_file} tidak ditemukan ❌")

def reset_indexed():
    indexed_file = "indexed.json"
    if os.path.exists(indexed_file):
        with open(indexed_file, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        print(f"Reset {indexed_file} ✅")
    else:
        print(f"{indexed_file} tidak ditemukan ❌")

if __name__ == "__main__":
    print("🚨 Resetting uploads and data...")
    reset_uploads()
    reset_chunks()
    reset_indexed()
    print("🎉 Reset complete!")