"""Khmer clean-up of greedy CTC output (karna.text.clean_khmer)."""

import pytest

from karna.text import clean_khmer


@pytest.mark.parametrize("raw, clean", [
    ("កាា", "កា"),                    # a vowel repeated: once
    ("មេាន", "មោន"),        # e + aa written apart: the vowel o
    ("កាុ", "កា"),                    # two vowels in a row: the first
    ("ក្ ខ", "ក ខ"),                  # a coeng before a space
    ("ាក ុខ", "ក ខ"),            # vowels opening a word
    ("ក្រុម", "ក្រុម"),  # valid text is kept
    ("ក  ខ ", "ក ខ"),                        # spaces
    ("hello  world", "hello world"),                               # English: spaces only
])
def test_clean_khmer(raw, clean):
    assert clean_khmer(raw) == clean
