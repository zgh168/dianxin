"""
Step 5: Inference on template.jsonl for submission.
CPU default, --gpu to use GPU (with cooldown).
"""
import argparse, json, os, time, torch, librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="Use GPU (default: CPU)")
    parser.add_argument("--model_path", default="./whisper-small")
    parser.add_argument("--input_jsonl", default="./repo_files/template.jsonl")
    parser.add_argument("--output_jsonl", default="./outputs/submission.jsonl")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--cooldown", type=float, default=0, help="Sleep seconds between samples (GPU: recommend 2)")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print("Loading model ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_path,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    print("Model loaded.", flush=True)

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        samples = samples[:args.limit]

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    print(f"Total: {len(samples)} samples", flush=True)
    t0 = time.time()
    results = []

    for i, sample in enumerate(samples):
        fname = sample["audio_path"]
        audio_path = os.path.join("repo_files", fname)

        if not os.path.isfile(audio_path):
            print(f"[{i+1}] SKIP: {audio_path}", flush=True)
            sample["pred_text"] = ""
            results.append(sample)
            continue

        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            feat = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
            feat = feat.to(device, dtype=model.dtype)

            with torch.no_grad():
                ids = model.generate(feat, max_new_tokens=128)
            pred = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

            sample["pred_text"] = pred
            results.append(sample)

            elapsed = time.time() - t0
            print(f"[{i+1}/{len(samples)}] {elapsed:.0f}s {pred[:60]}", flush=True)

            if args.cooldown > 0:
                time.sleep(args.cooldown)

        except Exception as e:
            print(f"[{i+1}] ERROR: {type(e).__name__}: {e}", flush=True)
            sample["pred_text"] = ""
            results.append(sample)

        if (i + 1) % args.save_every == 0 or i >= len(samples) - 1:
            with open(args.output_jsonl, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  saved {len(results)} results", flush=True)

    elapsed = time.time() - t0
    print(f"Done. {len(results)} samples, {elapsed:.0f}s ({elapsed/len(results):.1f}s/sample)")
    print(f"Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
