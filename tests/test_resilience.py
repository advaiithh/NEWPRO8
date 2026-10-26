"""A failure inside one turn must never end the live session."""
import time
from dataclasses import replace

import numpy as np

from pv.audio.io import load_wav16k
from pv.audio.playback import FilePlayer
from pv.pipeline import Features, Pipeline
from pv.runtime import Runtime
from pv.vad.silero_onnx import FRAME


def drive(p, samples, t0=0.0):
    t = t0
    for i in range(0, len(samples) - FRAME + 1, FRAME):
        t += FRAME / 16000
        p.process_frame(t, samples[i:i + FRAME])
    return t


def test_failed_accurate_pass_falls_back_to_fast_transcript(ollama_up, make_wav, tmp_path):
    rt = Runtime(ram_mb=8192)
    p = Pipeline(rt, FilePlayer(tmp_path / "r.wav"), replace(Features.full(), accurate_asr="base.en"), mode="always")
    try:
        class Boom:
            def transcribe(self, *a, **k):
                raise RuntimeError("mkl_malloc: failed to allocate memory")
        p.whisper_acc = Boom()
        p.strict = False
        p.turns = []
        p.on_turn = p.turns.append
        q = load_wav16k(make_wav("What is seventeen times twenty three?", "q"))
        t = drive(p, np.concatenate([np.zeros(16000, np.int16), q, np.zeros(16000, np.int16)]))
        for _ in range(100):
            if p.turns:
                break
            time.sleep(0.1)
            t = drive(p, np.zeros(FRAME * 4, np.int16), t)
        tl = [x for x in p.turns if x.path not in ("ignored", "empty")][0]
        assert "391" in tl.reply and "whisper_error" in tl.extra
    finally:
        p.close()
        rt.stop()


def test_exception_in_a_turn_is_contained_when_not_strict(ollama_up, tmp_path):
    rt = Runtime(ram_mb=8192)
    p = Pipeline(rt, FilePlayer(tmp_path / "r.wav"), Features.full(), mode="always")
    try:
        p.strict = False
        p._process_frame = lambda *a: (_ for _ in ()).throw(MemoryError("boom"))      # any failure inside a turn
        p.process_frame(0.0, np.zeros(FRAME, np.int16))                                # must not raise
        assert p.state == "idle" and p.tl is None
        p.strict = True
        try:
            p.process_frame(0.0, np.zeros(FRAME, np.int16))
            raise AssertionError("strict mode must re-raise")
        except MemoryError:
            pass
    finally:
        p.close()
        rt.stop()
