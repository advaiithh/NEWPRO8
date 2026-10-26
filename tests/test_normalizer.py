from pv.tts.normalizer import normalize, ordinal


def test_numbers_and_currency():
    assert normalize("₹450") == "four hundred fifty rupees"
    assert normalize("$1,250.50") == "one thousand two hundred fifty dollars and fifty cents"
    assert normalize("Rs 1") == "one rupee"


def test_percent_units_temperature():
    assert normalize("45%") == "forty five percent"
    assert normalize("12 km") == "twelve kilometers"
    assert normalize("2.5 kg") == "two point five kilograms"
    assert normalize("23°C") == "twenty three degrees Celsius"
    assert normalize("-5°C") == "minus five degrees Celsius"


def test_dates_times_ordinals():
    assert normalize("2026-10-08") == "October eighth, twenty twenty six"
    assert normalize("3:05") == "three oh five"
    assert normalize("the 21st") == "the twenty first"
    assert ordinal(2) == "second" and ordinal(40) == "fortieth"


def test_markdown_stripped():
    assert normalize("**Bold** and `code` here") == "Bold and code here"
    assert normalize("- one\n- two") == "one two"
