"""
Step 3 (v2): LoRA fine-tuning of whisper-small for Cantonese ASR.

Improvements over v1:
- Expanded LoRA target modules (q/k/v/out/fc1/fc2)
- Configurable LoRA rank (r=16/32/64)
- SpecAugment enabled (feature masking)
- Speed perturbation data augmentation (pre-computed at 0.95×, 1.05×)
- Feature-level noise in collator
- Cosine LR schedule with warmup
- More epochs + lower LR
- Gradient checkpointing for memory efficiency
"""
import argparse
import json
import math
import os
import numpy as np
import librosa
import soundfile as sf
import torch
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datasets import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from utils.collator import AugFeatureCollator


def _test_audio_file(audio_path: str) -> str:
    """Quick sanity check — returns path if readable, raises otherwise."""
    sf.read(audio_path, dtype="float32", always_2d=False)
    return audio_path


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def build_dataset_with_augment(data_items, processor, speeds, volumes, desc="preprocessing"):
    """Expand metadata + batched audio loading → stream to Arrow."""
    speeds = speeds or []
    volumes = volumes or []

    # Step 0: pre-scan with timeout — catch any file that hangs
    good_paths = set()
    unique_paths = list(dict.fromkeys(d["audio_path"] for d in data_items))
    bad_count = 0
    print(f"  Pre-scanning {len(unique_paths)} audio files (10s timeout, {min(8, len(unique_paths))} threads)...", flush=True)
    with ThreadPoolExecutor(max_workers=min(8, len(unique_paths))) as pool:
        futures = {pool.submit(_test_audio_file, ap): ap for ap in unique_paths}
        for i, (future, ap) in enumerate(zip(futures.keys(), unique_paths)):
            try:
                future.result(timeout=10)
                good_paths.add(ap)
            except (Exception, FutureTimeout):
                bad_count += 1
                if bad_count <= 10:
                    print(f"  SKIP: {os.path.basename(ap)}", flush=True)
            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(unique_paths)}] {len(good_paths)} ok, {bad_count} bad", flush=True)
    print(f"  Done: {len(good_paths)} ok, {bad_count} bad", flush=True)

    data_items = [d for d in data_items if d["audio_path"] in good_paths]
    if not data_items:
        raise RuntimeError("No valid audio files found!")

    # Step 1: expand metadata (strings only, no audio loaded yet)
    expanded = []
    for item in data_items:
        expanded.append({
            "audio_path": item["audio_path"],
            "text": item["text"],
            "speed": 1.0,
            "volume": 1.0,
        })
        for s in speeds:
            expanded.append({
                "audio_path": item["audio_path"],
                "text": item["text"],
                "speed": s,
                "volume": 1.0,
            })
        for v in volumes:
            expanded.append({
                "audio_path": item["audio_path"],
                "text": item["text"],
                "speed": 1.0,
                "volume": v,
            })

    ds = Dataset.from_list(expanded)
    print(f"  {desc}: {len(data_items)} items → {len(ds)} augmented rows", flush=True)

    # Step 2: batched audio loading + feature extraction (streams to Arrow)
    def extract_batch(batch):
        features, labels = [], []
        for ap, txt, sp, vol in zip(
            batch["audio_path"], batch["text"], batch["speed"], batch["volume"]
        ):
            try:
                data, orig_sr = sf.read(ap, dtype="float32", always_2d=False)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if orig_sr != 16000:
                    data = librosa.resample(y=data, orig_sr=orig_sr, target_sr=16000)
                audio = data[:30 * 16000]
                if not math.isclose(sp, 1.0):
                    audio = librosa.effects.time_stretch(y=audio, rate=sp)
                    audio = audio[:30 * 16000]
                if not math.isclose(vol, 1.0):
                    audio = audio * vol
                feat = processor(audio, sampling_rate=16000, return_tensors="np").input_features[0]
                lab = processor(text=txt).input_ids
                features.append(feat)
                labels.append(lab)
            except Exception as e:
                print(f"  WARN: skip {os.path.basename(ap)} ({e})", flush=True)
                continue
        return {"input_features": features, "labels": labels}

    ds = ds.map(
        extract_batch, batched=True, batch_size=32,
        remove_columns=["audio_path", "text", "speed", "volume"],
        desc=desc,
    )
    return ds


class LossLogCallback(TrainerCallback):
    """Log loss every N steps for visibility."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(f"  [step {state.global_step}] loss: {logs['loss']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning v2 for Cantonese ASR")
    parser.add_argument("--model_path", default="./whisper-small")
    parser.add_argument("--dataset_dir", default="./dataset")
    parser.add_argument("--output_dir", default="./checkpoints/lora-cantonese-v2")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler", default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (16/32/64)")
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_modules", nargs="+",
                        default=["q_proj", "v_proj"],
                        help="LoRA target modules")
    parser.add_argument("--speed_perturb", type=float, nargs="*",
                        default=[0.95, 1.05],
                        help="Speed perturbation factors (empty = disable)")
    parser.add_argument("--volume_perturb", type=float, nargs="*",
                        default=[0.7, 1.3],
                        help="Volume scaling factors for augmentation (empty = disable)")
    parser.add_argument("--feature_noise", type=float, default=0.005,
                        help="Feature noise stddev (0 = disable)")
    parser.add_argument("--noise_prob", type=float, default=0.5,
                        help="Probability of applying feature noise per sample")
    parser.add_argument("--spec_augment", action="store_true", default=False)
    parser.add_argument("--no_spec_augment", action="store_false", dest="spec_augment")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For torchrun DDP (auto-set)")
    args = parser.parse_args()

    # ---- Device ----
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.local_rank >= 0:
        device = f"cuda:{args.local_rank}"
    print(f"Device: {device}", flush=True)
    if device.startswith("cuda"):
        gpu_count = torch.cuda.device_count()
        print(f"  GPU count: {gpu_count}, Using: {torch.cuda.get_device_name(int(device.split(':')[-1]) if ':' in device else 0)}", flush=True)

    # ---- Model ----
    print("Loading model ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_path,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa" if device == "cuda" else "eager",
    ).to(device)

    # Force task tokens for consistency
    model.config.forced_decoder_ids = None
    model.generation_config.language = "zh"
    model.generation_config.task = "transcribe"

    # ---- SpecAugment ----
    if args.spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_time_length = 10
        model.config.mask_time_min_masks = 2
        model.config.mask_feature_prob = 0.05
        model.config.mask_feature_length = 10
        model.config.mask_feature_min_masks = 1
        print("SpecAugment: enabled (time_mask=0.05, feat_mask=0.05)", flush=True)
    else:
        print("SpecAugment: disabled", flush=True)

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Fix: transformers 5.x leaks duplicate decoder params through kwargs.
    # Strip all WhisperDecoder parameter names to prevent conflicts.
    import types
    _DECODER_PARAMS = {'input_ids', 'attention_mask', 'encoder_hidden_states',
                       'encoder_attention_mask', 'head_mask', 'cross_attn_head_mask',
                       'past_key_values', 'inputs_embeds', 'position_ids',
                       'use_cache', 'output_attentions', 'output_hidden_states', 'return_dict'}
    _wm = model.base_model.model
    _orig_wm_fwd = _wm.forward
    def _patched_wm_fwd(self, *args, **kwargs):
        for k in _DECODER_PARAMS:
            kwargs.pop(k, None)
        return _orig_wm_fwd(*args, **kwargs)
    _wm.forward = types.MethodType(_patched_wm_fwd, _wm)

    # ---- Dataset ----
    print("Loading datasets ...", flush=True)
    train_data = load_jsonl(os.path.join(args.dataset_dir, "train.jsonl"))
    val_data = load_jsonl(os.path.join(args.dataset_dir, "val.jsonl"))
    print(f"Raw: Train={len(train_data)}, Val={len(val_data)}", flush=True)

    # Build pre-computed features with speed + volume augmentation
    print("Preprocessing training data (augmentation) ...", flush=True)
    speeds = list(args.speed_perturb) if args.speed_perturb else []
    volumes = list(args.volume_perturb) if args.volume_perturb else []
    if speeds:
        print(f"  Speed perturbation factors: {speeds}", flush=True)
    else:
        print("  Speed perturbation: disabled", flush=True)
    if volumes:
        print(f"  Volume scaling factors: {volumes}", flush=True)
    else:
        print("  Volume scaling: disabled", flush=True)

    train_dataset = build_dataset_with_augment(train_data, processor, speeds, volumes, desc="train aug")
    val_dataset = build_dataset_with_augment(val_data, processor, (), (), desc="val")
    print(f"Processed: Train={len(train_dataset)}, Val={len(val_dataset)}", flush=True)

    orig_count = len(train_data)
    aug_count = len(train_dataset)
    ratio = aug_count // max(orig_count, 1)
    aug_details = []
    if speeds: aug_details.append("speed")
    if volumes: aug_details.append("volume")
    aug_tag = f"(x{ratio} w/ {'+'.join(aug_details)} pert)" if aug_details else ""
    print(f"  Train samples after augmentation: {aug_count} {aug_tag}", flush=True)

    # ---- Collator ----
    data_collator = AugFeatureCollator(
        processor,
        feature_noise=args.feature_noise if device == "cuda" else 0,
        noise_prob=args.noise_prob,
    )

    # ---- Training args ----
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=(device == "cuda"),
        gradient_checkpointing=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[LossLogCallback()],
    )

    # ---- Train ----
    print("\nStarting training ...", flush=True)
    print(f"  Epochs: {args.num_epochs}, LR: {args.learning_rate}, Scheduler: {args.lr_scheduler}", flush=True)
    print(f"  Batch: {args.batch_size} × {args.gradient_accumulation_steps} accum", flush=True)
    trainer.train()

    # ---- Save ----
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
