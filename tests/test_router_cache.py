"""D1: paraphrases hit the cache, out-of-domain misses, the margin rule, and skill-before-cache ordering."""
import pytest

from pv.router import skills
from pv.router.cache import SemanticCache, clean
from pv.router.router import Router


@pytest.fixture(scope="module")
def cache():
    return SemanticCache()


def test_paraphrases_hit_the_right_intent(cache):
    for text, intent in [("hello there", "greeting"), ("tell me something funny", "tell_joke"),
                         ("thanks so much", "thanks"), ("are you connected to wifi", "offline"),
                         ("how is the weather today", "weather")]:
        hit, sim, margin = cache.lookup(text)
        assert hit and hit.intent == intent, (text, sim, margin)
        assert hit.sim >= cache.tau and (hit.margin >= cache.margin or hit.sim >= 0.97)


def test_out_of_domain_misses(cache):
    for text in ["what is the capital of france", "who invented the telephone", "explain quantum entanglement",
                 "why does ice float on water"]:
        assert cache.lookup(text)[0] is None, text


def test_margin_rule_rejects_ambiguous_matches(cache):
    text = "hello hello anybody home"
    strict, lax = SemanticCache(), SemanticCache()
    strict.margin, lax.margin = 0.5, 0.0
    assert strict.lookup(text, tau=0.5)[0] is None            # a huge margin can never be met
    assert lax.lookup(text, tau=0.5)[0] is not None


def test_threshold_is_respected(cache):
    text = "make me smile with a joke"
    _, sim, _ = cache.lookup(text)
    assert cache.lookup(text, tau=sim + 0.01)[0] is None


def test_hindi_intents_only_match_hindi(cache):
    assert cache.lookup("नमस्ते", "hi")[0].intent == "namaste"
    assert cache.lookup("नमस्ते", "en")[0] is None


def test_clean_keeps_devanagari():
    assert clean("Hello, World!") == "hello world"
    assert clean("आप कैसे हैं?") == "आप कैसे हैं"


def test_router_prefers_skill_then_cache_then_llm(cache, tmp_path):
    ctx = skills.SkillContext.create(tmp_path)
    try:
        r = Router(ctx, cache)
        assert r.route("what is seventeen times twenty three").kind == "skill"
        c = r.route("tell me something funny")
        assert c.kind == "cache" and c.sim >= cache.tau
        assert r.route("why is the sky blue").kind == "llm"
        assert Router(ctx, cache, use_skills=False).route("what is 2 plus 2").kind == "llm"
        assert Router(ctx, cache, use_cache=False).route("tell me something funny").kind == "llm"
        assert r.intent_known("what time is it") and not r.intent_known("tell me about volcanoes")
    finally:
        ctx.timers.stop()


def test_cached_audio_is_prerendered():
    from pv.router.cache import CacheHit
    audio = CacheHit("greeting", 0, "x", 1.0, 1.0, "en").audio()
    assert audio is not None and len(audio[0]) > 3000


def test_tamil_cache_matches_tamil_and_not_english(cache):
    hit = cache.lookup("வணக்கம்", "ta")[0]
    assert hit and hit.intent == "ta_greeting" and "வணக்கம்" in hit.text
    assert cache.lookup("உங்கள் பெயர் என்னவென்று சொல்லுங்கள்", "ta")[0].intent == "ta_name"
    assert cache.lookup("இந்தியாவின் தலைநகரம் எது", "ta")[0] is None         # an unrelated question must not get a canned answer
    assert cache.lookup("வணக்கம்", "en")[0] is None
    assert cache.lookup("hello there", "ta")[0] is None


def test_tamil_and_hindi_audio_prerendered():
    from pv.router.cache import CacheHit
    for lang, intent in (("ta", "ta_greeting"), ("hi", "namaste")):
        audio = CacheHit(intent, 0, "x", 1.0, 1.0, lang).audio()
        assert audio is not None and len(audio[0]) > 3000, lang
