"""S1: Hindi and Tamil. Sticky language mode (voice command / key / flag) with Whisper-small recognition, per-language cache
(character similarity), per-language voice. Hindi is tested-partial (see docs/DEVIATIONS.md D-12); Tamil is measured in bench/tamil_eval.py."""
from dataclasses import replace

import soundfile as sf

from pv.asr import langid
from pv.audio.playback import FilePlayer
from pv.config import config, model
from pv.pipeline import Features, Pipeline
from pv.runtime import Runtime
from pv.tts.piper_tts import PiperTTS

TAMIL = lambda s: any("஀" <= c <= "௿" for c in s)       # noqa: E731


def speak(path, voice_name, text, scale=1.0):
    v = PiperTTS(voice_name, threads=2, length_scale=scale)
    sf.write(path, v.synth(text, do_normalize=False), v.sample_rate)


def make_pipe(tmp_path, **feat):
    rt = Runtime(ram_mb=8192)
    return rt, Pipeline(rt, FilePlayer(tmp_path / "r.wav"), replace(Features.full(), **feat), mode="ptt")


def test_hindi_mode_uses_the_hindi_voice(ollama_up, tmp_path):
    hindi = next(model("tts").glob("hi_IN-*.onnx")).stem
    speak(tmp_path / "hi.wav", hindi, "आप कैसे हैं")
    rt, p = make_pipe(tmp_path)
    try:
        p.set_language("hi")
        tl = p.run_wav(tmp_path / "hi.wav")
        assert tl.lang == "hi" and tl.reply.strip()
        assert tl.path == "cache" or p.last_voice.startswith("hi_IN"), (tl.path, p.last_voice)
    finally:
        p.close()
        rt.stop()


def test_tamil_mode_answers_in_tamil_with_the_tamil_voice(ollama_up, tmp_path):
    ta = config["tts"]["voice_ta"]
    speak(tmp_path / "how.wav", ta, "நீங்கள் எப்படி இருக்கிறீர்கள்")
    speak(tmp_path / "open.wav", ta, "இந்தியாவின் தலைநகரம் எது")
    rt, p = make_pipe(tmp_path)
    try:
        p.set_language("ta")
        tl = p.run_wav(tmp_path / "how.wav")                       # canned Tamil answer, pre-rendered Tamil audio
        assert tl.lang == "ta" and tl.path == "cache" and tl.extra["intent"] == "ta_how_are_you", (tl.path, tl.transcript, tl.extra)
        assert TAMIL(tl.reply)
        tl = p.run_wav(tmp_path / "open.wav")                      # open question: LLM reply spoken with the Tamil voice
        assert tl.lang == "ta" and tl.reply.strip() and p.last_voice.startswith("ta_IN"), (tl.transcript, p.last_voice)
    finally:
        p.close()
        rt.stop()


def test_voice_command_switches_language_both_ways(ollama_up, tmp_path):
    speak(tmp_path / "en_cmd.wav", "en_US-lessac-medium", "Speak Tamil.")
    speak(tmp_path / "ta_cmd.wav", config["tts"]["voice_ta"], "ஆங்கிலத்தில் பேசுங்கள்")
    rt, p = make_pipe(tmp_path)
    try:
        assert p.language == "en"
        tl = p.run_wav(tmp_path / "en_cmd.wav")
        assert p.language == "ta" and tl.path == "command" and tl.reply == langid.CONFIRM["ta"], (tl.transcript, tl.path)
        tl = p.run_wav(tmp_path / "ta_cmd.wav")
        assert p.language == "en" and tl.path == "command" and tl.reply == langid.CONFIRM["en"], (tl.transcript, tl.path)
    finally:
        p.close()
        rt.stop()


def test_english_is_the_default_and_unaffected(ollama_up, make_wav, tmp_path):
    rt, p = make_pipe(tmp_path)
    try:
        tl = p.run_wav(make_wav("Do I need internet to talk to you?", "en"))
        assert tl.lang == "en" and tl.path == "cache" and p.whisper is None       # the multilingual model was never loaded
    finally:
        p.close()
        rt.stop()


def test_optional_auto_detection_finds_tamil(ollama_up, tmp_path):
    speak(tmp_path / "how.wav", config["tts"]["voice_ta"], "நீங்கள் எப்படி இருக்கிறீர்கள்")
    rt, p = make_pipe(tmp_path, auto_language=True)
    try:
        tl = p.run_wav(tmp_path / "how.wav")
        assert tl.lang == "ta", (tl.lang, tl.transcript)
    finally:
        p.close()
        rt.stop()


def test_switch_command_parser():
    assert langid.switch_command("speak Tamil") == "ta" and langid.switch_command("Please switch to Hindi") == "hi"
    assert langid.switch_command("talk in English") == "en" and langid.switch_command("தமிழில் பேசு") == "ta"
    assert langid.switch_command("what is the capital of tamil nadu") is None
    assert langid.switch_command("tell me a story about a tamil king who spoke english to everyone") is None


def test_switch_command_survives_recogniser_slips_and_ignores_normal_questions():
    assert langid.switch_command("ஆங்கிரத்தில் பேசன் கழ்.") == "en"            # garbled "speak in English" heard in a live test
    assert langid.switch_command("Tamil") == "ta" and langid.switch_command("हिंदी में बोलो") == "hi"
    for normal in ("what is tamil nadu", "who is the president of india", "play some hindi songs", "tell me about the english language",
                   "நீங்கள் எப்படி இருக்கிறீர்கள்", "இந்தியாவின் தலைநகரம் எது", "ஒரு நகைச்சுவை சொல்லுங்கள்"):
        assert langid.switch_command(normal) is None, normal
