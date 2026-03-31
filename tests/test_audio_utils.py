from __future__ import annotations

import numpy as np

from app.audio.audio_utils import float32_to_pcm16


def test_float32_to_pcm16_flattens_tensor_like_audio():
    audio = np.array([[0.0, 0.5, -0.5]], dtype=np.float32)
    pcm_audio = float32_to_pcm16(audio)
    assert len(pcm_audio) == 6
