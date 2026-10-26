"""R5: VAD separates speech from silence; the wake word fires on its phrase and not on other speech."""
import numpy as np

from pv.audio.io import load_wav16k
from pv.audio.playback import resample
from pv.vad.silero_onnx import FRAME, SileroVAD
from pv.wake.oww import WakeWord


def voiced_fraction(x: np.ndarray) -> float:
    vad = SileroVAD()
    p = [vad.prob(x[i:i + FRAME]) for i in range(0, len(x) - FRAME, FRAME)]
    return float(np.mean(np.array(p) > vad.threshold))


def test_vad_speech_vs_silence(make_wav):
    speech = load_wav16k(make_wav("What is the capital of France and why is it famous?", "speech"))
    assert voiced_fraction(speech) > 0.7
    assert voiced_fraction(np.zeros(16000 * 2, np.int16)) == 0.0
    noise = (np.random.default_rng(0).normal(0, 300, 16000 * 2)).astype(np.int16)
    assert voiced_fraction(noise) < 0.1


def test_vad_is_fast():
    import time
    vad = SileroVAD()
    f = np.zeros(FRAME, np.int16)
    for _ in range(10):
        vad.prob(f)
    t = time.perf_counter()
    for _ in range(200):
        vad.prob(f)
    assert (time.perf_counter() - t) / 200 < 0.002           # target is <1 ms; allow 2 ms on a busy laptop


def peak_score(w: WakeWord, x: np.ndarray) -> float:
    w.reset()
    best = 0.0
    for i in range(0, len(x), FRAME):
        w.process(x[i:i + FRAME])
        best = max(best, w.last_score)
    return best


def test_wake_word_fires_on_phrase_not_on_other_speech(make_wav):
    w = WakeWord()
    pad = np.zeros(16000, np.int16)
    hey = np.concatenate([pad, load_wav16k(make_wav("Hey Jarvis", "hey")), pad])
    other = np.concatenate([pad, load_wav16k(make_wav("What is the capital of France and why is it famous?", "q")), pad])
    assert peak_score(w, hey) >= w.threshold
    assert peak_score(w, other) < w.threshold
    assert peak_score(w, np.zeros(16000 * 3, np.int16)) < w.threshold
