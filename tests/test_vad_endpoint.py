from pv.vad.endpoint import Endpointer, hangover_ms


def test_extends_for_incomplete_phrases():
    assert hangover_ms("what is the weather and", False) == 700
    assert hangover_ms("tell me about,", False) == 700
    assert hangover_ms("hello", False) == 700          # fewer than 2 words


def test_shrinks_for_complete_or_known_intent():
    assert hangover_ms("what time is it", False) < 350
    assert hangover_ms("what time is it", True) == 200
    assert hangover_ms("set a timer for five minutes", True) == 200


def test_default_without_partial():
    assert hangover_ms("", False) == 350
    assert hangover_ms("the sky looks quite blue today", False) == 350


def run(ep: Endpointer, partial: str, intent: bool = False, speech_frames: int = 20) -> tuple[float, float]:
    """Feed speech then silence; return (silence ms until end, last-voice time)."""
    t, ev = 0.0, None
    for _ in range(speech_frames):
        ev = ep.update(True, t, partial, intent)
        t += 0.032
    last = t - 0.032
    for n in range(1, 100):
        ev = ep.update(False, t, partial, intent)
        t += 0.032
        if ev and ev.kind == "end":
            assert abs(ev.t - last) < 1e-9
            return n * 32.0, ev.hangover_ms
    raise AssertionError("no end event")


def test_endpointer_adapts_end_to_end():
    quick, h1 = run(Endpointer(), "what time is it", intent=True)
    base, h2 = run(Endpointer(), "the sky looks quite blue today")
    slow, h3 = run(Endpointer(), "what is the weather and")
    assert (h1, h2, h3) == (200, 350, 700)
    assert quick < base < slow
    assert quick <= 224 and slow >= 700


def test_fixed_mode_ignores_partial():
    _, h = run(Endpointer(adaptive=False, base=500), "what is the weather and")
    assert h == 500


def test_short_blip_is_not_speech():
    ep = Endpointer()
    assert ep.update(True, 0.0) is None
    assert ep.update(False, 0.032) is None
    assert not ep.in_speech
