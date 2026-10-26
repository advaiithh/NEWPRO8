"""D5 barge-in and D4 live tightening, through the real pipeline (synthetic interruptions, no microphone)."""
import numpy as np

from pv.audio.io import load_wav16k
from pv.audio.playback import FilePlayer
from pv.pipeline import Features, Pipeline
from pv.runtime import Runtime
from pv.vad.silero_onnx import FRAME


def feed(p, samples, t0=0.0):
    t = t0
    for i in range(0, len(samples) - FRAME + 1, FRAME):
        t += FRAME / 16000
        p.process_frame(t, samples[i:i + FRAME])
    return t


def test_barge_in_cancels_reply_and_starts_new_turn(ollama_up, make_wav, tmp_path):
    from pv.router.skills.words import words_to_digits  # noqa: F401 (import check only)
    rt = Runtime(ram_mb=8192)
    p = Pipeline(rt, FilePlayer(tmp_path / "o.wav"), Features.full(), mode="ptt")
    try:
        done = []
        p.on_turn = done.append
        long_q = load_wav16k(make_wav("Explain how photosynthesis works in plants in some detail.", "long"))
        interrupt = load_wav16k(make_wav("Stop. What time is it right now?", "int"))
        p.trigger_wake()
        t = feed(p, np.concatenate([np.zeros(4800, np.int16), long_q, np.zeros(16000, np.int16)]))
        assert p.state == "busy", p.state                      # LLM is generating the long answer
        feed(p, np.concatenate([interrupt, np.zeros(16000 * 2, np.int16)]), t)
        for _ in range(300):
            if done and p.state == "idle":
                break
            import time
            time.sleep(0.05)
            feed(p, np.zeros(FRAME * 2, np.int16), t)
        assert any(tl.extra.get("barge_in") for tl in done), [tl.extra for tl in done]
        assert p.abort.is_set() or p.state == "idle"
    finally:
        p.close()
        rt.stop()


LIVE_TIGHTEN = """
import json, sys
import soundfile as sf
from pathlib import Path
from pv.audio.playback import FilePlayer
from pv.governor.governor import Governor
from pv.governor.tiers import build
from pv.pipeline import Features, Pipeline
from pv.runtime import Runtime
from pv.tts.piper_tts import PiperTTS

tmp = Path(sys.argv[1])
tts = PiperTTS("en_US-lessac-medium", threads=2, length_scale=1.0)
for name, text in (("a", "Why is the sky blue?"), ("b", "What is seventeen times twenty three?")):
    sf.write(tmp / (name + ".wav"), tts.synth(text), tts.sample_rate)
rt = Runtime()
holder = {}
gov = Governor(build(), rt.limiter, rt.sampler, lambda i: holder["p"].set_tier(i))
p = Pipeline(rt, FilePlayer(tmp / "t.wav"), Features.full(), tier=gov.tier_id, mode="ptt")
holder["p"], p.governor = p, gov
before = p.run_wav(tmp / "a.wav")
t0 = gov.tier_id
rt.limiter.tighten(2, 1536)
gov.tick()
after = p.run_wav(tmp / "b.wav")
print(json.dumps(dict(t0=t0, tier_after=gov.tier_id, reason=gov.log[-1]["reason"] if gov.log else "", before_tier=before.tier,
                      before_reply=bool(before.reply), after_tier=after.tier, after_reply=after.reply, ram=rt.limiter.ram_mb,
                      cores=rt.limiter.cores, cap=rt.limiter.job_limit_mb())))
"""


def test_live_tighten_drops_tier_and_assistant_keeps_answering(ollama_up, tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-c", LIVE_TIGHTEN, str(tmp_path)], cwd=root, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["t0"] == 0 and out["before_tier"] == 0 and out["before_reply"]
    assert out["tier_after"] >= 3 and out["reason"]                    # governor dropped tiers because the limit tightened
    assert out["after_tier"] == out["tier_after"] and out["after_reply"].strip()      # still answers at the lower tier, no crash
    assert out["ram"] == 1536 and out["cores"] == 2 and out["cap"] >= 1536
