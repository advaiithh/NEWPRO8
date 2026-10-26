"""R1: a spoken question goes in, a spoken answer comes out, fully offline, with a populated timeline."""
import soundfile as sf

from pv.guard import netguard
from pv.metrics.timeline import STAMPS


def test_baseline_end_to_end(ollama_up, make_wav, tmp_path):
    from pv.audio.playback import FilePlayer
    from pv.baseline import Baseline
    from pv.runtime import Runtime

    netguard.reset()
    rt = Runtime(ram_mb=8192, split=False)
    out = tmp_path / "reply.wav"
    b = Baseline(rt, FilePlayer(out))
    tl = b.run_wav(make_wav("What is the capital of France?", "q"))
    rt.stop()
    assert "france" in tl.transcript.lower()
    assert tl.reply.strip() and out.exists()
    audio, sr = sf.read(out)
    assert len(audio) / sr > 1.0                                  # a spoken answer, not silence
    for k in ("speech_end", "asr_final", "llm_first_token", "first_audio", "turn_end"):
        assert k in tl.stamps, k
    assert tl.latency_ms and tl.latency_ms > 800                  # includes the fixed 800 ms hangover
    assert netguard.status()["attempts"] == 0
    assert set(tl.stamps) <= set(STAMPS)


def _pipeline(tmp_path, mode="ptt"):
    from pv.audio.playback import FilePlayer
    from pv.pipeline import Features, Pipeline
    from pv.runtime import Runtime
    rt = Runtime(ram_mb=8192)
    return rt, Pipeline(rt, FilePlayer(tmp_path / "ours.wav"), Features.full(), mode=mode)


def test_streaming_pipeline_all_three_routes(ollama_up, make_wav, tmp_path):
    netguard.reset()
    rt, p = _pipeline(tmp_path)
    try:
        got = {}
        for name, text in [("skill", "What time is it right now?"), ("cache", "Do I need internet to talk to you?"),
                           ("llm", "Why is the sky blue?")]:
            tl = p.run_wav(make_wav(text, name))
            got[name] = tl
            assert tl.path == name, (name, tl.path, tl.transcript)
            assert tl.reply and (tmp_path / "ours.wav").exists()
            for k in ("speech_end", "asr_final", "route_decision", "tts_first_chunk_ready", "first_audio", "turn_end"):
                assert k in tl.stamps, (name, k)
            assert tl.tier == 0 and tl.hangover_ms > 0 and tl.cpu_s > 0
        assert got["skill"].latency_ms < 1500 and got["cache"].latency_ms < 1500
        assert got["llm"].latency_ms < 4000
        assert got["llm"].tokens > 0 and "llm_first_token" in got["llm"].stamps
        assert netguard.status()["attempts"] == 0
    finally:
        p.close()
        rt.stop()


def test_wake_word_starts_a_turn_without_push_to_talk(ollama_up, make_wav, tmp_path):
    import numpy as np
    from pv.audio.io import WavSource, load_wav16k
    rt, p = _pipeline(tmp_path, mode="wake")
    try:
        hey = load_wav16k(make_wav("Hey Jarvis", "hey"))
        q = load_wav16k(make_wav("What is seventeen times twenty three?", "q17"))
        audio = np.concatenate([hey, np.zeros(int(0.5 * 16000), np.int16), q])
        done = []
        p.on_turn = done.append
        for t, frame in WavSource(audio, tail_s=4.0).frames():
            p.process_frame(t, frame)
            if p.turn_done.is_set():
                break
        assert done, "wake word did not start a turn"
        tl = done[0]
        assert "wake" in tl.stamps and tl.stamps["wake"] < tl.stamps["speech_end"]
        assert "391" in tl.reply and tl.path == "skill"
    finally:
        p.close()
        rt.stop()
