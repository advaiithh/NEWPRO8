"""Tamil end to end on synthetic speech: Tamil voice -> Whisper small -> cache. Reports recognition CER and cache hit / false-hit
rates at several thresholds (text-only and after speech recognition)."""
from __future__ import annotations

import json

import jiwer
import numpy as np

from bench.common import OUT
from pv.asr.whisper_asr import WhisperASR
from pv.audio.playback import resample
from pv.config import config
from pv.router.cache import SemanticCache
from pv.tts.piper_tts import PiperTTS

POS = [("வணக்கம் நண்பரே", "ta_greeting"), ("நீங்கள் நலமாக இருக்கிறீர்களா", "ta_how_are_you"),
       ("உங்கள் பெயர் என்னவென்று சொல்லுங்கள்", "ta_name"), ("மிக்க நன்றி உங்களுக்கு", "ta_thanks"),
       ("நான் போய் வருகிறேன்", "ta_goodbye"), ("எனக்கு ஒரு ஜோக் சொல்லுங்கள்", "ta_joke"),
       ("இனிய காலை வணக்கம் உங்களுக்கு", "ta_good_morning"), ("நான் இப்போது தூங்கப் போகிறேன்", "ta_good_night"),
       ("நீங்கள் என்ன செய்ய முடியும் சொல்லுங்கள்", "ta_capabilities"), ("இணைய இணைப்பு இல்லாமல் வேலை செய்யுமா", "ta_offline"),
       ("நீங்கள் தமிழ் பேசுவீர்களா", "ta_language"), ("எனக்கு உதவி தேவைப்படுகிறது", "ta_help")]
NEG = ["இந்தியாவின் தலைநகரம் எது", "இன்று வானிலை எப்படி இருக்கும்", "சூரியன் எங்கே உதிக்கிறது", "ஒளிச்சேர்க்கை என்றால் என்ன",
       "தமிழ்நாட்டின் முதலமைச்சர் யார்", "பூமி எவ்வளவு பெரியது", "இந்த திரைப்படம் எப்போது வெளியானது", "சென்னையில் எத்தனை மக்கள் வாழ்கிறார்கள்"]


def rates(cache, texts_pos, texts_neg, tau):
    hit = wrong = 0
    for t, intent in texts_pos:
        h, _, _ = cache.lookup(t, "ta", tau)
        hit += h is not None
        wrong += h is not None and h.intent != intent
    fp = sum(cache.lookup(t, "ta", tau)[0] is not None for t in texts_neg)
    return {"tau": tau, "hit": hit / len(texts_pos), "wrong_of_hits": wrong / hit if hit else 0.0, "false_hit": fp / len(texts_neg)}


def main() -> None:
    cache = SemanticCache()
    tts = PiperTTS(config["tts"]["voice_ta"], threads=2, length_scale=1.0)
    asr = WhisperASR(config["asr"]["multilingual"], threads=4, fast=True)
    heard_pos, heard_neg, cer = [], [], []
    for t, intent in POS:
        text = asr.transcribe(resample(tts.synth(t, do_normalize=False), tts.sample_rate, 16000), "ta")[0]
        heard_pos.append((text, intent))
        cer.append(jiwer.cer(t, text))
    for t in NEG:
        heard_neg.append(asr.transcribe(resample(tts.synth(t, do_normalize=False), tts.sample_rate, 16000), "ta")[0])
    taus = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
    res = {"cer_mean": float(np.mean(cer)), "text_only": [rates(cache, POS, NEG, t) for t in taus],
           "after_recognition": [rates(cache, heard_pos, heard_neg, t) for t in taus], "heard_examples": heard_pos[:4]}
    (OUT / "tamil_eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"recognition CER on paraphrases: {res['cer_mean']:.2f}")
    for k in ("text_only", "after_recognition"):
        for r in res[k]:
            print(f"{k:18} tau={r['tau']:.2f} hit={r['hit']:.2f} wrong/hit={r['wrong_of_hits']:.2f} false-hit={r['false_hit']:.2f}")


if __name__ == "__main__":
    main()
