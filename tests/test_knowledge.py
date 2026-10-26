"""Offline knowledge base: definition routing, grounded extracts, and refusal to trust look-alike articles."""
import pytest

from pv.knowledge.kb import Hit, Knowledge, topic_of


@pytest.fixture(scope="module")
def kb():
    return Knowledge()


def test_topic_extraction():
    assert topic_of("What is photosynthesis?") == ("photosynthesis", True)
    assert topic_of("who was Mahatma Gandhi") == ("mahatma gandhi", True)
    assert topic_of("tell me about the Eiffel Tower") == ("eiffel tower", True)
    assert topic_of("what is the capital of Japan")[1] is False        # a fact about a topic, not a definition
    assert topic_of("why is the sky blue")[1] is False


def test_definitions_are_read_from_the_article(kb):
    for q, title in [("what is photosynthesis", "Photosynthesis"), ("who is Albert Einstein", "Albert Einstein"),
                     ("what is a black hole", "Black hole"), ("tell me about elephants", "Elephant")]:
        kind, hit, text = kb.answer(q)
        assert kind == "extract" and hit.title == title, (q, kind, hit and hit.title)
        assert 5 <= len(text.split()) <= 50


def test_lookalike_articles_are_not_trusted(kb):
    kind, hit, _ = kb.answer("who was the first president of the United States")
    assert kind != "context" or "president" in hit.title.lower() or "united states" in hit.title.lower()
    kind, hit, text = kb.answer("what is diabetes")
    assert not (kind == "extract" and hit.title == "American Diabetes Association")


def test_named_topic_gives_context_for_a_fact_question(kb):
    kind, hit, text = kb.answer("who wrote Romeo and Juliet")
    assert kind == "context" and hit.title == "Romeo and Juliet" and "Shakespeare" in text


def test_lookup_is_fast(kb):
    import time
    kb.answer("what is the capital of Japan")
    t = time.perf_counter()
    for _ in range(20):
        kb.answer("who painted the Mona Lisa")
    assert (time.perf_counter() - t) / 20 < 0.15


def test_snippet_is_short():
    h = Hit("X", "First sentence is here. Second sentence is also here. Third one. " + "word " * 100, 1.0, True)
    assert len(h.snippet().split()) <= 45


# ---- the current (October 2026) English Wikipedia knowledge base --------------------------------------------------
import pytest  # noqa: E402

from pv import config as cfgmod  # noqa: E402
from pv.config import model as _model  # noqa: E402
from pv.knowledge.kb import office_topic  # noqa: E402

needs_full_kb = pytest.mark.skipif(not (_model("kb_en") / "emb.npy").exists(), reason="run scripts/build_kb_full.py first")


def test_office_question_parsing_expands_abbreviations():
    assert office_topic("who is the current CM of Tamil Nadu") == "chief minister of tamil nadu"
    assert office_topic("Who is the Prime Minister of India?") == "prime minister of india"
    assert office_topic("who is the president of the united states right now") == "president of the united states"
    assert office_topic("who is Albert Einstein") is None and office_topic("who wrote hamlet") is None


@pytest.fixture(scope="module")
def full_kb():
    cfgmod.reload("quality")
    try:
        yield Knowledge()
    finally:
        cfgmod.reload(None)


@needs_full_kb
def test_full_kb_is_the_october_2026_snapshot(full_kb):
    assert full_kb.snapshot == "20261001" and full_kb.asof == "October 2026" and len(full_kb.titles) > 300000


@needs_full_kb
@pytest.mark.parametrize("question, must_contain", [
    ("who is the current CM of Tamil Nadu", ["Vijay", "chief minister of Tamil Nadu"]),
    ("who is the prime minister of India", ["Modi", "prime minister of India"]),
    ("who is the president of the United States", ["Trump", "president of the United States"]),
    ("who is the president of India", ["Murmu", "president of India"]),
    ("who is the prime minister of the United Kingdom", ["Burnham", "Prime Minister of the United Kingdom"]),
    ("who is the secretary general of the United Nations", ["Guterres", "secretary-general of the United Nations"]),
    ("who is the chancellor of Germany", ["Merz", "Chancellor of Germany"]),
    ("who is the chief minister of West Bengal", ["Suvendu Adhikari", "Chief Minister of West Bengal"]),
    ("who is the prime minister of Japan", ["Takaichi", "Prime Minister of Japan"]),
])
def test_current_office_holders_come_from_the_october_2026_data(full_kb, question, must_contain):
    """Expected names are what the October 2026 snapshot says (some changed in 2026: Tamil Nadu, West Bengal, UK, Japan)."""
    kind, hit, text = full_kb.answer(question)
    assert kind == "extract" and text.startswith("According to Wikipedia as of October 2026:"), (question, kind, text)
    for part in must_contain:
        assert part.lower() in text.lower(), (question, text)


@needs_full_kb
def test_wrong_entities_are_not_accepted_as_the_holder(full_kb):
    """Regressions found while building this: Air Force One, a Principal Secretary, a party president, a past term."""
    for q, bad in [("who is the president of the United States", "Air Force One"), ("who is the prime minister of India", "Principal Secretary"),
                   ("who is the president of India", "Majlis"), ("who is the prime minister of the United Kingdom", "Questions")]:
        assert bad not in full_kb.answer(q)[2], q


@needs_full_kb
def test_election_results_need_a_named_year_and_the_right_event(full_kb):
    kind, hit, text = full_kb.answer("who won the 2026 Tamil Nadu election")
    assert kind == "extract" and hit.title == "2026 Tamil Nadu Legislative Assembly election" and "2026" in text
    kind, hit, text = full_kb.answer("who won the 2026 West Bengal election")
    assert hit.title == "2026 West Bengal Legislative Assembly election"
    assert full_kb.answer("who won the world cup")[0] != "extract"            # no edition named: do not guess


@needs_full_kb
def test_full_kb_lookup_stays_fast(full_kb):
    import time
    full_kb.answer("who is the current CM of Tamil Nadu")
    t = time.perf_counter()
    for q in ("who is the prime minister of india", "what is photosynthesis", "who painted the mona lisa"):
        full_kb.answer(q)
    assert (time.perf_counter() - t) / 3 < 1.0
