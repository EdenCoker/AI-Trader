"""Story-identity guarantees: edit variants merge, distinct events stay apart."""

from __future__ import annotations

from ai_trader.news.identity import (
    SIMILARITY_THRESHOLD,
    cluster_titles,
    normalize_title,
    similarity,
    story_id_for,
    story_vector,
)

# (a, b, same_story) — the labeled pair set the threshold is tuned on.
LABELED_PAIRS = [
    (
        "Apple beats Q3 earnings expectations as iPhone sales surge",
        "Apple tops Q3 earnings expectations on surging iPhone sales - Reuters",
        True,
    ),
    (
        "Fed raises interest rates by 25 basis points",
        "Federal Reserve raises rates by 25 basis points",
        True,
    ),
    (
        "Microsoft to acquire Activision Blizzard for $69 billion",
        "Microsoft acquires Activision Blizzard in $69 billion deal",
        True,
    ),
    (
        "Nvidia shares jump after record data center revenue",
        "Nvidia stock surges on record data-center revenue",
        True,
    ),
    ("Turkey hikes rates to 50%", "Argentina hikes rates to 50%", False),
    (
        "Apple beats Q3 earnings expectations",
        "Amazon beats Q3 earnings expectations",
        False,
    ),
    (
        "US imposes 12th sanctions package on Russia",
        "China imposes new tariffs on US goods",
        False,
    ),
    (
        "Tesla recalls 2 million vehicles over autopilot",
        "Toyota recalls 1 million vehicles over airbags",
        False,
    ),
    (
        "Boeing wins $10 billion order from Emirates",
        "Airbus wins $12 billion order from Qatar Airways",
        False,
    ),
    (
        "Oil prices rise on OPEC supply cuts",
        "Gold prices rise on safe-haven demand",
        False,
    ),
]

MARGIN = 0.04


def test_labeled_pairs_fully_separate_with_margin():
    for a, b, same in LABELED_PAIRS:
        sim = similarity(story_vector(a), story_vector(b))
        if same:
            assert sim >= SIMILARITY_THRESHOLD + MARGIN, (a, b, sim)
        else:
            assert sim <= SIMILARITY_THRESHOLD - MARGIN, (a, b, sim)


def test_normalize_strips_source_suffix():
    assert normalize_title("Fed raises rates - Reuters") == "fed raises rates"
    assert normalize_title("Fed raises rates | CNBC") == "fed raises rates"


def test_normalize_collapses_finance_aliases():
    assert normalize_title("Federal Reserve raises rates") == "fed raises rates"


def test_identical_titles_have_identical_ids():
    a = story_id_for(normalize_title("Apple beats estimates - Reuters"))
    b = story_id_for(normalize_title("Apple beats estimates"))
    assert a == b


def test_empty_title_is_unvectorizable():
    assert story_vector("") is None
    assert story_vector("!!! ...") is None


def test_clustering_merges_variants_and_keeps_singletons():
    titles = [
        "Apple beats Q3 earnings expectations as iPhone sales surge",
        "Apple tops Q3 earnings expectations on surging iPhone sales - Reuters",
        "Fed raises interest rates by 25 basis points",
        "Turkey hikes rates to 50%",
        "",
    ]
    clusters = cluster_titles(titles)
    assert [0, 1] in clusters
    assert [2] in clusters
    assert [3] in clusters
    assert [4] in clusters  # unvectorizable stays singleton, no pooled identity


def test_cjk_titles_vectorize_via_char_bigrams():
    vec = story_vector("美联储加息25个基点")
    assert vec is not None
    same = story_vector("美联储加息25个基点")
    assert similarity(vec, same) > 0.99
