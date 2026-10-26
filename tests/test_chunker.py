from pv.tts.chunker import Chunker


def run(tokens: list[str]) -> list[str]:
    c, out = Chunker(), []
    for t in tokens:
        out += c.push(t)
    return out + c.flush()


def stream(text: str) -> list[str]:
    return run([w + " " for w in text.split()])


def test_first_chunk_is_short_and_ends_at_clause():
    out = stream("The capital of France is Paris, which is also its largest city. It is lovely.")
    assert out[0] == "The capital of France is Paris,"
    assert 4 <= len(out[0].split()) <= 10


def test_sentence_boundary_and_flush():
    out = stream("Yes. It is sunny today. Bye")
    assert out == ["Yes. It is sunny today.", "Bye"]


def test_forced_cut_without_punctuation():
    out = stream("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen")
    assert len(out[0].split()) == 10 and all(len(o.split()) <= 12 for o in out)


def test_decimals_and_abbreviations_not_split():
    out = stream("Dr. Smith paid 3.5 dollars for it today.")
    assert out == ["Dr. Smith paid 3.5 dollars for it today."]
