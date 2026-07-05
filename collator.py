"""
Data collator with on-the-fly feature augmentation for Whisper LoRA training.

Applies:
- Feature-level Gaussian noise
- Works alongside model-level SpecAugment
"""
import torch


class AugFeatureCollator:
    """Pad input_features and labels; optionally inject feature noise."""

    def __init__(self, processor, feature_noise: float = 0.005, noise_prob: float = 0.5):
        self.processor = processor
        self.feature_noise = feature_noise
        self.noise_prob = noise_prob

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        labels = [{"input_ids": f["labels"]} for f in features]

        feat_batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_batch = self.processor.tokenizer.pad(labels, return_tensors="pt", padding=True)

        # Debug: show all keys from feature_extractor.pad()
        if not hasattr(self, '_debug_done'):
            print(f"  [DEBUG] feature_extractor.pad() keys: {list(feat_batch.keys())}", flush=True)
            print(f"  [DEBUG] tokenizer.pad() keys: {list(label_batch.keys())}", flush=True)

        # Newer transformers returns (B, 1, mel, time) — squeeze to (B, mel, time)
        feats = feat_batch["input_features"]
        if feats.dim() == 4:
            feats = feats.squeeze(1)

        # Feature-level augmentation
        if self.feature_noise > 0 and self.noise_prob > 0:
            noise_mask = torch.rand(len(features), 1, 1, device=feats.device) < self.noise_prob
            noise = torch.randn_like(feats) * self.feature_noise
            feats = feats + (noise * noise_mask)

        # Labels with -100 for padding
        labs = label_batch["input_ids"].masked_fill(
            label_batch["attention_mask"].ne(1), -100
        )

        # Build clean dict — ONLY the keys model expects, nothing else
        result = {"input_features": feats, "labels": labs}

        if not hasattr(self, '_debug_done'):
            print(f"  [DEBUG] final batch keys: {list(result.keys())}", flush=True)
            print(f"  [DEBUG] input_features shape: {result['input_features'].shape}", flush=True)
            print(f"  [DEBUG] labels shape: {result['labels'].shape}", flush=True)
            self._debug_done = True

        return result
