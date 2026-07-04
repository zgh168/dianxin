"""
Step 3: LoRA fine-tuning of whisper-small for Cantonese ASR using HuggingFace Trainer.
"""
import argparse, json, os, torch
from datasets import Dataset
from transformers import (
    AutoProcessor, AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model
import librosa
import numpy as np


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def preprocess_function(examples, processor):
    """Process batch for Whisper - audio as input_features, text as labels."""
    input_features_list = []
    labels_list = []

    for audio_path, text in zip(examples["audio_path"], examples["text"]):
        # Load and resample audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio = audio[:30 * 16000]  # max 30s

        # Extract features
        feat = processor(audio, sampling_rate=sr, return_tensors="np").input_features[0]
        input_features_list.append(feat)

        # Tokenize text as labels
        labels = processor(text=text).input_ids
        labels_list.append(labels)

    return {
        "input_features": input_features_list,
        "labels": labels_list,
    }


class DataCollator:
    """Pad input_features and labels to batch max length."""
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        labels = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(labels, return_tensors="pt", padding=True)

        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./whisper-small")
    parser.add_argument("--dataset_dir", default="./dataset")
    parser.add_argument("--output_dir", default="./checkpoints/lora-cantonese")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load processor and model
    print("Loading model ...")
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_path,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    print("Loading datasets ...")
    train_data = load_jsonl(os.path.join(args.dataset_dir, "train.jsonl"))
    val_data = load_jsonl(os.path.join(args.dataset_dir, "val.jsonl"))
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Preprocess
    print("Preprocessing (this may take a few minutes) ...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, processor),
        batched=True,
        batch_size=32,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, processor),
        batched=True,
        batch_size=32,
        remove_columns=val_dataset.column_names,
    )
    print("Preprocessing done.")

    data_collator = DataCollator(processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=(device == "cuda"),
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=processor.tokenizer,
    )

    print("Starting training ...")
    trainer.train()

    # Save final model
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
