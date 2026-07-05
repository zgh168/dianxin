"""
Audio augmentation utilities for ASR training.

Provides:
- Raw audio augmentation: speed, noise, volume, equalization
- Feature-level augmentation: SpecAugment-compatible noise
- Augmented data collator for on-the-fly feature perturbation
"""
import random
import numpy as np
import librosa


def speed_perturb(audio: np.ndarray, sr: int = 16000, speed: float = None) -> np.ndarray:
    """Perturb audio speed. If speed is None, randomly pick from 0.9–1.1."""
    if speed is None:
        speed = random.uniform(0.9, 1.1)
    audio = librosa.effects.time_stretch(y=audio, rate=speed)
    return audio


def volume_perturb(audio: np.ndarray, factor: float = None) -> np.ndarray:
    """Random volume scaling (0.7–1.3)."""
    if factor is None:
        factor = random.uniform(0.7, 1.3)
    return audio * factor


def add_gaussian_noise(audio: np.ndarray, level: float = None) -> np.ndarray:
    """Add Gaussian noise. level ~ 0.001–0.01."""
    if level is None:
        level = random.uniform(0.001, 0.01)
    noise = np.random.randn(len(audio)) * level
    return audio + noise


def add_short_noise_burst(audio: np.ndarray, sr: int = 16000, prob: float = 0.2) -> np.ndarray:
    """Add a short random noise burst at a random position."""
    if random.random() >= prob:
        return audio
    burst_len = random.randint(int(0.01 * sr), int(0.05 * sr))  # 10–50ms
    burst_level = random.uniform(0.01, 0.05)
    pos = random.randint(0, max(1, len(audio) - burst_len))
    noise_burst = np.random.randn(burst_len) * burst_level
    audio = audio.copy()
    audio[pos:pos + burst_len] += noise_burst
    return audio


def augment_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Full augmentation pipeline for one audio clip. Applied with ~30% probability each."""
    if random.random() < 0.3:
        audio = speed_perturb(audio, sr)
        audio = audio[:30 * sr]  # re-trim after speed change
    if random.random() < 0.5:
        audio = volume_perturb(audio)
    if random.random() < 0.3:
        audio = add_gaussian_noise(audio)
    if random.random() < 0.2:
        audio = add_short_noise_burst(audio, sr)
    return audio


def extract_augmented_features(
    audio_path: str,
    text: str,
    processor,
    speeds: list = None,
    volumes: list = None,
    max_duration: int = 30,
    sr: int = 16000,
):
    """
    Load audio, extract features at original + speed-perturbed + volume-scaled versions.
    Returns list of dicts: [{"input_features": ..., "labels": ...}, ...]
    """
    if speeds is None:
        speeds = []
    if volumes is None:
        volumes = []
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    audio = audio[:max_duration * sr]

    results = []

    def _extract(waveform):
        feat = processor(waveform, sampling_rate=sr, return_tensors="np").input_features[0]
        labels = processor(text=text).input_ids
        results.append({"input_features": feat, "labels": labels})

    # Original version
    _extract(audio)

    # Speed-perturbed copies
    for speed in speeds:
        aug_audio = librosa.effects.time_stretch(y=audio, rate=speed)
        aug_audio = aug_audio[:max_duration * sr]
        _extract(aug_audio)

    # Volume-scaled copies (on original-speed audio)
    for vol in volumes:
        _extract(audio * vol)

    return results


def prepare_augmented_dataset(data_items, processor, speeds=(0.95, 1.05), volumes=(0.7, 1.3)):
    """Build lists of features+labels with speed + volume augmentation."""
    import sys
    all_features, all_labels = [], []
    total = len(data_items)
    multi = 1 + len(speeds) + len(volumes)
    print(f"  Preprocessing {total} samples x{multi} augment = {total * multi} features ...", flush=True)
    for i, item in enumerate(data_items):
        results = extract_augmented_features(
            item["audio_path"], item["text"], processor,
            speeds=list(speeds), volumes=list(volumes),
        )
        for r in results:
            all_features.append(r["input_features"])
            all_labels.append(r["labels"])
        if (i + 1) % 50 == 0 or i < 3:
            print(f"  [{i+1}/{total}]", flush=True)
    print(f"  Done: {len(all_features)} features", flush=True)
    return all_features, all_labels
