"""Story identity: "are these two headlines the same news story?"

One similarity definition for the whole package — dedup, corroboration
counting, and story tracking all call into here, so there is exactly one
answer to the question (multiple inconsistent similarity definitions are
how corroboration counts silently diverge from what the UI/dedup shows).

Method (original implementation; the design approach — feature-hashed
lexical vectors with an entity/number boost, scored as the min of two
independently seeded views — is informed by the architecture notes of the
open-source worldmonitor project):

- word tokens (weight 2.0) carry core lexical identity
- word bigrams (weight 1.5) encode order, separating actor-flipped
  headlines that bag-of-words cannot ("ukraine drone" vs "russian drone")
- character 4-grams per token (weight 1.0) absorb morphology
  (iran/iranian, sanction/sanctions)
- tokens that were capitalized in the raw headline get a 3x boost and
  numeric tokens a 2x boost: entities and magnitudes carry the
  discriminating signal, so "Turkey hikes rates to 50%" and "Argentina
  hikes rates to 50%" stay separate stories

Features are hashed (FNV-1a, signed) into a fixed-dimension vector and
L2-normalized. Similarity is the MINIMUM of the two views' cosines: a
hash collision would have to happen in both independently seeded views to
produce a false merge. Deterministic, dependency-free, microseconds per
headline.

This is an EDIT-TOLERANT identity, not a semantic one. It merges wording
variants (source suffixes, truncations, reorders) and keeps distinct
events apart; it cannot merge a cross-language paraphrase.
"""

from __future__ import annotations

import hashlib
import math
import re

DIM = 512
# Tuned on the labeled pairs in tests/test_news_identity.py; retune there
# if the vectorizer changes.
SIMILARITY_THRESHOLD = 0.62

_WEIGHT_TOKEN = 2.0
_WEIGHT_BIGRAM = 1.5
_WEIGHT_CHARGRAM = 0.75
_BOOST_ENTITY = 3.0
_BOOST_NUMBER = 2.0

# Finance-domain aliases collapsed BEFORE tokenizing, so common
# abbreviation pairs ("Fed" / "Federal Reserve") don't fork a story.
# Longest-first replacement; keep entries lowercase.
_ALIASES: tuple[tuple[str, str], ...] = (
    ("federal reserve", "fed"),
    ("european central bank", "ecb"),
    ("bank of england", "boe"),
    ("bank of japan", "boj"),
    ("international monetary fund", "imf"),
    ("securities and exchange commission", "sec"),
    ("department of justice", "doj"),
    ("federal trade commission", "ftc"),
    ("initial public offering", "ipo"),
    ("mergers and acquisitions", "m a"),
    ("artificial intelligence", "ai"),
)
_ALIAS_RES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(r"\b" + re.escape(alias) + r"\b"), short) for alias, short in _ALIASES
)


def _apply_aliases(text_lower: str) -> str:
    for pattern, short in _ALIAS_RES:
        text_lower = pattern.sub(short, text_lower)
    return text_lower

_VIEW_SEEDS = (0x9E3779B1, 0x85EBCA77)

# Trailing source attributions ("... - Reuters", "... | CNBC") fork
# identity if left in place; strip before vectorizing.
_SOURCE_SUFFIX_RE = re.compile(r"\s+[-|–—]\s+[A-Za-z][\w .&']{1,40}$")
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_title(title: str) -> str:
    """Lowercase, strip trailing source attribution and non-word noise,
    collapse whitespace. The normalized form is the hashing identity for
    canonical story ids."""

    text = _apply_aliases(_SOURCE_SUFFIX_RE.sub("", title.strip()).lower())
    tokens = _TOKEN_RE.findall(text)
    return " ".join(tokens)


def story_id_for(normalized_title: str) -> str:
    return hashlib.sha256(normalized_title.encode("utf-8")).hexdigest()[:24]


def _fnv1a(data: str, seed: int) -> int:
    h = (0x811C9DC5 ^ seed) & 0xFFFFFFFF
    for ch in data:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _is_numberish(token: str) -> bool:
    return any(ch.isdigit() for ch in token)


def _features(title: str) -> list[tuple[str, float]]:
    stripped = _SOURCE_SUFFIX_RE.sub("", title.strip())
    raw_tokens = _TOKEN_RE.findall(stripped)
    if not raw_tokens:
        return []

    # Alias substitution runs on the lowercase text, so token positions can
    # shift; recover per-token capitalization by checking membership of the
    # original capitalized token set instead of positional pairing.
    capitalized = {token.lower() for token in raw_tokens if token[:1].isupper()}
    lowered = _TOKEN_RE.findall(_apply_aliases(stripped.lower()))
    if not lowered:
        return []

    features: list[tuple[str, float]] = []
    for token in lowered:
        weight = _WEIGHT_TOKEN
        # Capitalized-in-raw tokens are entity-shaped; numeric tokens are
        # event parameters. The boosts do NOT stack: a capitalized number
        # ("Q3", "50%") is a parameter shared across many stories, and
        # stacking let its shared mass drown the actual entity difference.
        # In Title Case or ALL-CAPS headlines every token gets the boost —
        # uniform scaling, which cosine ignores — so this only sharpens
        # sentence case.
        if _is_numberish(token):
            weight *= _BOOST_NUMBER
        elif token in capitalized:
            weight *= _BOOST_ENTITY
        features.append((f"t:{token}", weight))

    for first, second in zip(lowered, lowered[1:], strict=False):
        features.append((f"b:{first} {second}", _WEIGHT_BIGRAM))

    for token in lowered:
        if token.isascii():
            if len(token) >= 4:
                for i in range(len(token) - 3):
                    features.append((f"c:{token[i : i + 4]}", _WEIGHT_CHARGRAM))
        else:
            # Unsegmented scripts (CJK) produce no useful word tokens;
            # character bigrams carry the signal instead.
            for i in range(len(token) - 1):
                features.append((f"c2:{token[i : i + 2]}", _WEIGHT_CHARGRAM))

    return features


def story_vector(title: str) -> tuple[tuple[float, ...], ...] | None:
    """Dual-view hashed vector for a headline, or None when the headline
    has no vectorizable content (empty / punctuation-only)."""

    features = _features(title)
    if not features:
        return None

    views: list[tuple[float, ...]] = []
    for seed in _VIEW_SEEDS:
        vec = [0.0] * DIM
        for feature, weight in features:
            index_hash = _fnv1a(feature, seed)
            sign_hash = _fnv1a(feature, seed ^ 0x5BD1E995)
            sign = 1.0 if sign_hash & 1 else -1.0
            vec[index_hash % DIM] += sign * weight
        norm = math.sqrt(sum(value * value for value in vec))
        if norm == 0:
            return None
        views.append(tuple(value / norm for value in vec))
    return tuple(views)


def similarity(
    a: tuple[tuple[float, ...], ...],
    b: tuple[tuple[float, ...], ...],
) -> float:
    """Min-of-views cosine similarity between two story vectors."""

    return min(
        sum(x * y for x, y in zip(view_a, view_b, strict=True))
        for view_a, view_b in zip(a, b, strict=True)
    )


def cluster_titles(titles: list[str], threshold: float = SIMILARITY_THRESHOLD) -> list[list[int]]:
    """Greedy single-pass clustering of headline indices.

    Each title joins the first existing cluster containing a member within
    ``threshold``; otherwise it starts a new cluster. Deterministic in
    input order. Unvectorizable titles become singleton clusters (they
    cannot be compared, so they must not merge or pool corroboration).
    """

    vectors = [story_vector(title) for title in titles]
    clusters: list[list[int]] = []
    for index, vector in enumerate(vectors):
        placed = False
        if vector is not None:
            for cluster in clusters:
                for member in cluster:
                    member_vector = vectors[member]
                    if member_vector is None:
                        continue
                    if similarity(vector, member_vector) >= threshold:
                        cluster.append(index)
                        placed = True
                        break
                if placed:
                    break
        if not placed:
            clusters.append([index])
    return clusters
