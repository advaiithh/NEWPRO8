import json
import time

import pytest

from pv.router import skills
from pv.router.skills.calc import safe_eval
from pv.router.skills.words import words_to_digits


@pytest.fixture
def ctx(tmp_path):
    c = skills.SkillContext.create(tmp_path)
    yield c
    c.timers.stop()


def ask(ctx, text):
    r = skills.run(text, ctx)
    return r[1] if r else None


def test_arithmetic(ctx):
    assert ask(ctx, "what is 17 times 23") == "That is 391."
    assert ask(ctx, "what is seventeen times twenty three") == "That is 391."
    assert ask(ctx, "what's 20 percent of 450?") == "That is 90."
    assert ask(ctx, "what is the square root of 144") == "That is 12."
    assert ask(ctx, "what is 10 divided by 4") == "That is 2.5."


def test_no_eval_injection(ctx):
    assert ask(ctx, "what is __import__('os').getcwd()") is None
    for bad in ["__import__('os')", "open('x')", "9**9**9", "(1).__class__", "lambda: 1"]:
        with pytest.raises((ValueError, SyntaxError)):
            safe_eval(bad)


def test_words_to_digits():
    assert words_to_digits("twenty three") == "23"
    assert words_to_digits("two hundred and five") == "205"
    assert words_to_digits("two and a half minutes") == "2.5 minutes"
    assert words_to_digits("one thousand two hundred") == "1200"
    assert words_to_digits("seventeen times twenty-three") == "17 times 23"


def test_units(ctx):
    assert ask(ctx, "convert 5 miles to kilometers") == "5 miles is 8.05 kilometers."
    assert ask(ctx, "how many centimeters in 3 inches") == "3 inches is 7.62 centimeters."
    assert ask(ctx, "100 fahrenheit in celsius") == "100 degrees fahrenheit is 37.78 degrees celsius."


def test_time_date_misc(ctx):
    assert ask(ctx, "what time is it").startswith("It is ")
    assert "Today is" in ask(ctx, "what's the date today")
    assert ask(ctx, "flip a coin") in ("It is heads.", "It is tails.")
    assert ask(ctx, "tell me about france") is None
    assert ask(ctx, "louder") == "Okay, louder." and ctx.actions[-1][0] == "volume"


def test_timer_set_list_and_persist(ctx, tmp_path):
    assert ask(ctx, "set a timer for 5 minutes") == "Okay, timer set for 5 minutes."
    assert "timer has" in ask(ctx, "how much time is left on my timer")
    again = skills.SkillContext.create(tmp_path)           # simulates restart
    try:
        assert len(again.timers.items) == 1
    finally:
        again.timers.stop()


def test_timer_fires(tmp_path):
    fired = []
    c = skills.SkillContext.create(tmp_path, on_timer=fired.append)
    try:
        c.timers.add(0.6, "test timer")
        time.sleep(1.6)
        assert fired == ["Your test timer is done."]
    finally:
        c.timers.stop()


def test_notes_persist(ctx, tmp_path):
    assert ask(ctx, "remember that the wifi password is blue horse") == "Okay, I will remember that the wifi password is blue horse."
    again = skills.SkillContext.create(tmp_path)
    try:
        assert "blue horse" in ask(again, "what did I ask you to remember")
    finally:
        again.timers.stop()
    assert json.loads((tmp_path / "notes.json").read_text())[0]["text"].startswith("the wifi")
