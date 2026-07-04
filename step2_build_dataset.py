"""
Step 2: Build training/validation dataset from index.csv + data/ audio files.
Outputs HuggingFace Dataset dict with train/val splits.
"""
import argparse
import os
import sys
import json
import csv
import glob

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def build_audio_mapping(data_root):
    """Build filename -> absolute path mapping for all .wav files in data/"""
    mapping = {}
    for root, dirs, files in os.walk(data_root):
        for f in files:
            if f.endswith('.wav') and not os.path.basename(f).startswith('._'):
                mapping[f] = os.path.join(root, f)
    return mapping


def load_index_csv(path):
    """Load index.csv, return list of {序号, 粤语原文, ...}"""
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = row.get("序号", "").strip()
            yueyu = row.get("粤语原文", "").strip()
            if seq_id and yueyu:
                rows.append({"seq_id": seq_id, "yueyu": yueyu})
    return rows


def build_seq_id_map(audio_map):
    """
    Build seq_id (int) -> filename mapping from audio filenames.
    Filenames like '00001你好！.wav' -> seq_id=1
    """
    id_map = {}
    for fname in audio_map:
        fbase = os.path.splitext(fname)[0]
        digits = ""
        for ch in fbase:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            id_map[int(digits)] = fname
    return id_map


def match_audio(rows, audio_map):
    """Match index rows with audio files using seq_id lookup."""
    print("Building seq_id index ...")
    seq_map = build_seq_id_map(audio_map)
    print(f"Indexed {len(seq_map)} audio files by seq_id")

    dataset = []
    matched = 0
    for row in rows:
        seq_id = int(row["seq_id"])
        if seq_id in seq_map:
            fname = seq_map[seq_id]
            dataset.append({
                "audio_path": audio_map[fname],
                "text": row["yueyu"],
            })
            matched += 1

    print(f"Matched {matched}/{len(rows)} rows")
    if matched == 0:
        # Debug: show sample seq_ids and map keys
        print(f"Sample row seq_ids: {[r['seq_id'] for r in rows[:5]]}")
        print(f"Sample map keys: {sorted(seq_map.keys())[:10]}")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./repo_files/data")
    parser.add_argument("--index_csv", default="./repo_files/index.csv")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", default="./dataset")
    args = parser.parse_args()

    print("Building audio mapping ...")
    audio_map = build_audio_mapping(args.data_root)
    print(f"Found {len(audio_map)} audio files")

    print("Loading index.csv ...")
    rows = load_index_csv(args.index_csv)
    print(f"Found {len(rows)} index entries")

    print("Matching audio with labels ...")
    dataset = match_audio(rows, audio_map)

    # Split train/val
    import random
    random.seed(42)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_size = int(len(dataset) * args.val_ratio)
    val_indices = set(indices[:val_size])
    train_indices = set(indices[val_size:])

    train_data = [d for i, d in enumerate(dataset) if i in train_indices]
    val_data = [d for i, d in enumerate(dataset) if i in val_indices]

    os.makedirs(args.output_dir, exist_ok=True)

    # Save as JSONL
    for split, data in [("train", train_data), ("val", val_data)]:
        path = os.path.join(args.output_dir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{split}: {len(data)} samples -> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
