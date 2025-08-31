import numpy as np
import librosa
from typing import Dict


def analyze_voice(wav_path: str) -> Dict:
    """Compute basic voice features: pitch stats, speaking rate proxy, energy."""
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    
    # Energy
    rms = librosa.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))

    # Pitch (F0) using librosa piptrack as a simple proxy
    S = np.abs(librosa.stft(y))
    pitches, magnitudes = librosa.piptrack(S=S, sr=sr)
    pitch_vals = pitches[magnitudes > np.median(magnitudes)]
    pitch_vals = pitch_vals[pitch_vals > 0]
    
    if len(pitch_vals) == 0:
        pitch_stats = {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    else:
        pitch_stats = {
            "mean": float(np.mean(pitch_vals)),
            "min": float(np.min(pitch_vals)),
            "max": float(np.max(pitch_vals)),
            "std": float(np.std(pitch_vals)),
        }

    # Pause proxy: proportion of low-energy frames
    thresh = np.percentile(rms, 20)
    pause_ratio = float(np.mean(rms < thresh))

    return {
        "energy_mean": energy_mean,
        "pitch": pitch_stats,
        "pause_ratio": pause_ratio,
    }
