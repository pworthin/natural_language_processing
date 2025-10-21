
import pytest
from Week07v1 import PocketTranslator, formatcheck

def test_batch_skips_long_phrases(capfd):
    # Generate one long string (1001 chars)
    too_long = "a" * 1001
    normal = "Hello, how are you?"

    batch = [normal, too_long, "This is fine."]

    # Create a translator instance
    translator = PocketTranslator(
        text=batch,
        src='en',
        src_fn='English',
        dest='fr',
        dest_fn='French'
    )

    results = translator.batch_translation(batch)

    # Capture printed output
    out, _ = capfd.readouterr()

    # Make sure the skip message was printed
    assert "too long" in out.lower()

    # Make sure the overlong phrase is not in the results
    assert all(len(entry['original']) <= 1000 for entry in results)

    # Should only return 2 translations
    assert len(results) == 2

    # Confirm correct original lines are in output
    originals = [r['original'] for r in results]
    assert normal in originals
    assert too_long not in originals




# Helper: simulate input and see if it returns None (error was caught)
def test_missing_delimiter():
    sc, sfn, dc, dfn = formatcheck("english spanish")
    assert sc is None  # Error expected, should return None

def test_numeric_input():
    sc, sfn, dc, dfn = formatcheck("123 > 456")
    assert sc is None  # Should catch non-alphabetical input

def test_unknown_language():
    sc, sfn, dc, dfn = formatcheck("Klingon > French")
    assert sc is None  # Should not find Klingon

def test_valid_single_language():
    sc, sfn, dc, dfn = formatcheck("english > german")
    assert sc == 'en'
    assert dc == ['de']

def test_valid_multiple_languages():
    sc, sfn, dc, dfn = formatcheck("english > french, spanish, japanese")
    assert sc == 'en'
    assert set(dc) == {'fr', 'es', 'ja'}



