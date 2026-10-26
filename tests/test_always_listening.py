"""Hands-free mode: no SPACE, no wake word. It answers only when the user speaks."""
import time
from dataclasses import replace

import numpy as np

from pv.audio.io import load_wav16k
from pv.audio.playback import FilePlayer
from pv.pipeline import Features, Pipeline
from pv.runtime import Runtime
from pv.vad.silero_onnx import FRAME


def drive(p, samples, t0=0.0, pace=False):
    t = t0
    for i in range(0, len(samples) - FRAME + 1, FRAME):
        t += FRAME / 16000
        p.process_frame(t, samples[i:i + FRAME])
        if pace:
            time.sleep(FRAME / 16000)
    return t


def make(tmp_path, barge_in=False):
    rt = Runtime(ram_mb=8192)
    p = Pipeline(rt, FilePlayer(tmp_path / "r.wav"), replace(Features.full(), barge_in=barge_in), mode="always")
    p.turns = []
    p.on_turn = p.turns.append
    return rt, p


def test_silence_and_noise_never_get_an_answer(ollama_up, tmp_path):
    rt, p = make(tmp_path)
    try:
        rng = np.random.default_rng(1)
        hum = (rng.normal(0, 60, 16000 * 4)).astype(np.int16)           # quiet fan / room noise
        loud = (rng.normal(0, 3000, 16000 * 3)).astype(np.int16)        # loud broadband noise (not speech)
        t = drive(p, np.zeros(16000 * 3, np.int16))
        t = drive(p, hum, t)
        t = drive(p, loud, t)
        drive(p, np.zeros(16000 * 3, np.int16), t)
        time.sleep(1.0)
        assert [x for x in p.turns if x.path not in ("ignored", "empty")] == []      # nothing was answered
    finally:
        p.close()
        rt.stop()


def test_answers_exactly_once_when_spoken_to_then_stays_quiet(ollama_up, make_wav, tmp_path):
    rt, p = make(tmp_path)
    try:
        q = load_wav16k(make_wav("What is seventeen times twenty three?", "q"))
        t = drive(p, np.zeros(16000 * 2, np.int16))
        t = drive(p, q, t)
        t = drive(p, np.zeros(16000, np.int16), t)
        for _ in range(100):
            if p.turns:
                break
            time.sleep(0.1)
            t = drive(p, np.zeros(FRAME * 4, np.int16), t)
        t = drive(p, np.zeros(16000 * 6, np.int16), t)               # then silence: must stay silent
        answered = [x for x in p.turns if x.path not in ("ignored", "empty")]
        assert len(answered) == 1 and "391" in answered[0].reply, [(x.path, x.reply) for x in p.turns]
        assert "wake" not in answered[0].stamps                       # no wake word / key involved
    finally:
        p.close()
        rt.stop()


def test_it_does_not_react_to_sound_while_it_is_replying(ollama_up, make_wav, tmp_path):
    rt, p = make(tmp_path, barge_in=False)                            # default for speaker users
    try:
        long_q = load_wav16k(make_wav("Explain how photosynthesis works in plants in some detail.", "long"))
        noise_speech = load_wav16k(make_wav("Stop. What time is it right now?", "int"))
        t = drive(p, np.concatenate([np.zeros(16000, np.int16), long_q, np.zeros(16000, np.int16)]))
        assert p.state == "busy"
        t = drive(p, noise_speech, t)                                 # sound arrives while it is answering
        for _ in range(200):
            if p.turns and p.state == "idle":
                break
            time.sleep(0.1)
            t = drive(p, np.zeros(FRAME * 4, np.int16), t)
        assert not any(x.extra.get("barge_in") for x in p.turns)
        first = [x for x in p.turns if x.path not in ("ignored", "empty")]
        assert len(first) == 1 and "photosynthesis" in first[0].transcript.lower()
    finally:
        p.close()
        rt.stop()


def test_junk_transcripts_are_not_answered():
    assert Pipeline._is_noise("") and Pipeline._is_noise("you") and Pipeline._is_noise("Uh.") and Pipeline._is_noise("the the")
    assert not Pipeline._is_noise("hello") and not Pipeline._is_noise("what time is it") and not Pipeline._is_noise("stop")
