"""
Word bank for the imposter game.

Each entry is (word, category). Words should be concrete enough to clue
without being trivially guessable from a single obvious clue.
"""

WORD_BANK: list[tuple[str, str]] = [
    # Animals
    ("penguin",     "animal"),
    ("elephant",    "animal"),
    ("octopus",     "animal"),
    ("flamingo",    "animal"),
    ("kangaroo",    "animal"),
    ("chameleon",   "animal"),
    ("hummingbird", "animal"),
    # Food & drink
    ("sushi",       "food"),
    ("cinnamon",    "food"),
    ("avocado",     "food"),
    ("espresso",    "food"),
    ("pretzel",     "food"),
    ("sourdough",   "food"),
    ("wasabi",      "food"),
    # Places
    ("lighthouse",  "place"),
    ("library",     "place"),
    ("submarine",   "place"),
    ("airport",     "place"),
    ("aquarium",    "place"),
    ("monastery",   "place"),
    ("skyscraper",  "place"),
    # Objects
    ("telescope",   "object"),
    ("compass",     "object"),
    ("hourglass",   "object"),
    ("microscope",  "object"),
    ("typewriter",  "object"),
    ("parachute",   "object"),
    ("kaleidoscope","object"),
    # Nature
    ("glacier",     "nature"),
    ("volcano",     "nature"),
    ("thunderstorm","nature"),
    ("coral reef",  "nature"),
    ("quicksand",   "nature"),
    ("aurora",      "nature"),
    # Concepts / activities
    ("meditation",  "concept"),
    ("camouflage",  "concept"),
    ("hibernation", "concept"),
    ("marathon",    "concept"),
    ("improvisation","concept"),
    ("archaeology", "concept"),
]

import random


def pick_word(category: str | None = None) -> tuple[str, str]:
    """Return a random (word, category) pair, optionally filtered by category."""
    pool = WORD_BANK if category is None else [w for w in WORD_BANK if w[1] == category]
    if not pool:
        raise ValueError(f"No words for category '{category}'")
    return random.choice(pool)


CATEGORIES = sorted({w[1] for w in WORD_BANK})
