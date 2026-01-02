"""
Utility functions for numeric encoding of drug names, ICD codes, and CPT codes.

These encoders are designed to:
- Produce **fully numeric** features (no one-hot over raw strings).
- Be fast enough for event-level data via vectorized pandas string ops.
- Be reusable across feature-engineering and final-model steps.

All functions are side-effect free and do not perform any I/O.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import math
from collections import Counter


def _safe_str_series(s: pd.Series) -> pd.Series:
    """Normalize a Series to string, treating NA as empty string."""
    return s.fillna("").astype(str)


_VOWELS_LOWER = set("aeiouy")
_VOWELS_UPPER = set("AEIOUY")


def _compute_positional_cv_features(
    s: pd.Series, prefix: str
) -> pd.DataFrame:
    """
    Positional / trend-based consonant-vowel features for a string Series.

    For each cleaned string:
      - Compute cv_slope from a simple linear regression of cv_signed on pos_norm.
      - Vowel ratios in first vs second half and their difference.
      - Centers of mass for vowels and consonants and their difference.
      - Simple binary flags for boundary patterns.

    Assumes strings are already normalized (case) appropriately for the domain.
    """
    def _features_for_one(text: str) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
        if not text:
            # cv_slope, vr1, vr2, vr_delta, center_v, center_c, center_diff,
            # starts_with_vowel, ends_with_vowel,
            # vowel_majority_first_half, vowel_majority_second_half,
            # vowel_run_end
            return (
                0.0,  # cv_slope
                0.0,  # vowel_ratio_first_half
                0.0,  # vowel_ratio_second_half
                0.0,  # vowel_ratio_delta
                0.5,  # center_of_vowels
                0.5,  # center_of_consonants
                0.0,  # center_diff
                0.0,  # starts_with_vowel
                0.0,  # ends_with_vowel
                0.0,  # vowel_majority_first_half
                0.0,  # vowel_majority_second_half
                0.0,  # has_vowel_run_end
            )

        chars = list(text)
        L = len(chars)
        # Precompute normalized positions
        if L == 1:
            x_vals = [0.0]
        else:
            denom = float(L - 1)
            x_vals = [i / denom for i in range(L)]

        # Build cv_signed and vowel/consonant positions
        cv_signed = []
        vowel_positions = []
        consonant_positions = []
        first_half_v = first_half_c = 0
        second_half_v = second_half_c = 0

        half_idx = L // 2

        for i, ch in enumerate(chars):
            is_alpha = ch.isalpha()
            is_vowel = (ch in _VOWELS_LOWER) or (ch in _VOWELS_UPPER)
            if is_alpha:
                if is_vowel:
                    cv_signed.append(1.0)
                    vowel_positions.append(x_vals[i])
                    if i < half_idx:
                        first_half_v += 1
                    else:
                        second_half_v += 1
                else:
                    cv_signed.append(-1.0)
                    consonant_positions.append(x_vals[i])
                    if i < half_idx:
                        first_half_c += 1
                    else:
                        second_half_c += 1
            else:
                cv_signed.append(0.0)

        # Linear regression slope of cv_signed ~ pos_norm
        n = float(L)
        Sx = sum(x_vals)
        Sy = sum(cv_signed)
        Sxx = sum(x * x for x in x_vals)
        Sxy = sum(x * y for x, y in zip(x_vals, cv_signed))
        denom_lr = n * Sxx - Sx * Sx
        if denom_lr != 0.0:
            cv_slope = (n * Sxy - Sx * Sy) / denom_lr
        else:
            cv_slope = 0.0

        # Vowel ratios first / second half
        first_total = first_half_v + first_half_c
        second_total = second_half_v + second_half_c
        if first_total > 0:
            vowel_ratio_first_half = first_half_v / float(first_total)
        else:
            vowel_ratio_first_half = 0.0
        if second_total > 0:
            vowel_ratio_second_half = second_half_v / float(second_total)
        else:
            vowel_ratio_second_half = 0.0
        vowel_ratio_delta = vowel_ratio_second_half - vowel_ratio_first_half

        # Centers of vowels / consonants in normalized position space
        if vowel_positions:
            center_of_vowels = sum(vowel_positions) / float(len(vowel_positions))
        else:
            center_of_vowels = 0.5
        if consonant_positions:
            center_of_consonants = sum(consonant_positions) / float(len(consonant_positions))
        else:
            center_of_consonants = 0.5
        center_diff = center_of_vowels - center_of_consonants

        # Boundary / run-based flags
        first_ch = chars[0]
        last_ch = chars[-1]
        starts_with_vowel = 1.0 if ((first_ch in _VOWELS_LOWER) or (first_ch in _VOWELS_UPPER)) else 0.0
        ends_with_vowel = 1.0 if ((last_ch in _VOWELS_LOWER) or (last_ch in _VOWELS_UPPER)) else 0.0

        vowel_majority_first_half = 1.0 if first_half_v > first_half_c and first_total > 0 else 0.0
        vowel_majority_second_half = 1.0 if second_half_v > second_half_c and second_total > 0 else 0.0

        # Last N=3 chars all vowels (simple choice for "vowel run" at end)
        N = 3
        if L >= N:
            tail = chars[-N:]
            has_vowel_run_end = 1.0 if all(
                (ch in _VOWELS_LOWER) or (ch in _VOWELS_UPPER) for ch in tail
            ) else 0.0
        else:
            has_vowel_run_end = ends_with_vowel

        return (
            float(cv_slope),
            float(vowel_ratio_first_half),
            float(vowel_ratio_second_half),
            float(vowel_ratio_delta),
            float(center_of_vowels),
            float(center_of_consonants),
            float(center_diff),
            float(starts_with_vowel),
            float(ends_with_vowel),
            float(vowel_majority_first_half),
            float(vowel_majority_second_half),
            float(has_vowel_run_end),
        )

    results = s.fillna("").astype(str).apply(_features_for_one)

    return pd.DataFrame(
        {
            f"{prefix}_cv_slope": results.apply(lambda t: t[0]),
            f"{prefix}_vowel_ratio_first_half": results.apply(lambda t: t[1]),
            f"{prefix}_vowel_ratio_second_half": results.apply(lambda t: t[2]),
            f"{prefix}_vowel_ratio_delta": results.apply(lambda t: t[3]),
            f"{prefix}_center_of_vowels": results.apply(lambda t: t[4]),
            f"{prefix}_center_of_consonants": results.apply(lambda t: t[5]),
            f"{prefix}_center_diff": results.apply(lambda t: t[6]),
            f"{prefix}_starts_with_vowel": results.apply(lambda t: t[7]),
            f"{prefix}_ends_with_vowel": results.apply(lambda t: t[8]),
            f"{prefix}_vowel_majority_first_half": results.apply(lambda t: t[9]),
            f"{prefix}_vowel_majority_second_half": results.apply(lambda t: t[10]),
            f"{prefix}_has_vowel_run_end": results.apply(lambda t: t[11]),
        }
    ).astype("float32")


def _compute_entropy_and_run_features(
    s: pd.Series, prefix: str
) -> pd.DataFrame:
    """
    Entropy, diversity, and consonant-run features for a string Series.

    For each cleaned string:
      - Character-level entropy and bigram diversity.
      - Consonant run statistics and vowel/consonant transitions on the
        alphabetic-only projection of the string.
    """
    def _metrics(text: str) -> Tuple[float, float, float, float, float, float, float, float]:
        if not text:
            # char_entropy, bigram_diversity, max_cons_run, mean_cons_run,
            # vc_transition_rate, tri_cons_clusters, vowel_runs_ge2, alpha_len
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        chars = list(text)
        L = len(chars)

        # Character entropy
        counts = Counter(chars)
        total = float(L)
        probs = [c / total for c in counts.values()]
        char_entropy = -sum(p * math.log2(p) for p in probs if p > 0.0)

        # Bigram diversity
        if L >= 2:
            bigrams = {text[i : i + 2] for i in range(L - 1)}
            bigram_diversity = len(bigrams) / float(L - 1)
        else:
            bigram_diversity = 0.0

        # Alphabetic-only projection
        alpha_chars = [ch for ch in chars if ch.isalpha()]
        A = len(alpha_chars)
        if A == 0:
            return (char_entropy, bigram_diversity, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Vowel/consonant types
        types = []
        for ch in alpha_chars:
            if (ch in _VOWELS_LOWER) or (ch in _VOWELS_UPPER):
                types.append("V")
            else:
                types.append("C")

        # Consonant runs
        cons_runs = []
        cur_run = 0
        for t in types:
            if t == "C":
                cur_run += 1
            else:
                if cur_run > 0:
                    cons_runs.append(cur_run)
                    cur_run = 0
        if cur_run > 0:
            cons_runs.append(cur_run)

        if cons_runs:
            max_cons_run = float(max(cons_runs))
            mean_cons_run = float(sum(cons_runs) / len(cons_runs))
        else:
            max_cons_run = 0.0
            mean_cons_run = 0.0

        # Vowel/consonant transitions
        if A > 1:
            transitions = sum(1 for i in range(A - 1) if types[i] != types[i + 1])
            vc_transition_rate = transitions / float(A - 1)
        else:
            vc_transition_rate = 0.0

        # Tri-consonant clusters
        tri_clusters = 0
        if A >= 3:
            for i in range(A - 2):
                if types[i] == types[i + 1] == types[i + 2] == "C":
                    tri_clusters += 1

        # Vowel runs length >= 2
        vowel_runs_ge2 = 0
        cur_v = 0
        for t in types:
            if t == "V":
                cur_v += 1
            else:
                if cur_v >= 2:
                    vowel_runs_ge2 += 1
                cur_v = 0
        if cur_v >= 2:
            vowel_runs_ge2 += 1

        return (
            float(char_entropy),
            float(bigram_diversity),
            float(max_cons_run),
            float(mean_cons_run),
            float(vc_transition_rate),
            float(tri_clusters),
            float(vowel_runs_ge2),
            float(A),
        )

    results = s.fillna("").astype(str).apply(_metrics)

    return pd.DataFrame(
        {
            f"{prefix}_char_entropy": results.apply(lambda t: t[0]),
            f"{prefix}_bigram_diversity": results.apply(lambda t: t[1]),
            f"{prefix}_max_consonant_run_len": results.apply(lambda t: t[2]),
            f"{prefix}_mean_consonant_run_len": results.apply(lambda t: t[3]),
            f"{prefix}_vc_transition_rate": results.apply(lambda t: t[4]),
            f"{prefix}_tri_consonant_cluster_count": results.apply(lambda t: t[5]),
            f"{prefix}_vowel_run_count_ge2": results.apply(lambda t: t[6]),
            f"{prefix}_alpha_len_for_runs": results.apply(lambda t: t[7]),
        }
    ).astype("float32")


# ---------------------------------------------------------------------------
# Drug name encoding
# ---------------------------------------------------------------------------


def encode_drug_name_series(drug_series: pd.Series, prefix: str = "drug") -> pd.DataFrame:
    """
    Encode raw drug name strings into a set of numeric features.

    This implements the core of the README_feature_encoding.md drug-name
    strategy using vectorized pandas string operations.
    """
    s = _safe_str_series(drug_series).str.strip().str.lower()

    # Basic character-level views
    no_space = s.str.replace(r"\s+", "", regex=True)
    alpha = no_space.str.replace("[^a-z]", "", regex=True)
    digits = no_space.str.replace("[^0-9]", "", regex=True)

    alpha_len = alpha.str.len()
    char_len = no_space.str.len()

    # Vowels / consonants (simple heuristic)
    vowel_count = alpha.str.count("[aeiouy]")
    consonant_count = alpha_len.sub(vowel_count)

    # Syllable estimate: count vowel groups
    syllable_count = s.str.replace("[^a-z]", " ", regex=True).str.count("[aeiouy]+")

    # Tokenization for token-level metrics
    tokens = s.str.split()
    token_count = tokens.apply(len)

    # Helper lambdas for per-row computations that are hard to vectorize
    def _max_token_len(tok_list) -> int:
        if not tok_list:
            return 0
        return max(len("".join([ch for ch in t if ch.isalpha()])) for t in tok_list)

    def _frac_alpha_tokens(tok_list) -> float:
        if not tok_list:
            return 0.0
        alpha_tok = sum(1 for t in tok_list if t.isalpha())
        return alpha_tok / len(tok_list)

    def _frac_alnum_tokens(tok_list) -> float:
        if not tok_list:
            return 0.0
        alnum_tok = sum(1 for t in tok_list if any(ch.isalpha() for ch in t) and any(ch.isdigit() for ch in t))
        return alnum_tok / len(tok_list)

    def _max_token_digit_count(tok_list) -> int:
        if not tok_list:
            return 0
        return max(sum(ch.isdigit() for ch in t) for t in tok_list)

    max_token_len = tokens.apply(_max_token_len)
    frac_alpha_tokens = tokens.apply(_frac_alpha_tokens)
    frac_alnum_tokens = tokens.apply(_frac_alnum_tokens)
    max_token_digit_count = tokens.apply(_max_token_digit_count)

    # Composition / ratios
    vowel_ratio = vowel_count.div(alpha_len.replace(0, pd.NA)).fillna(0.0)
    consonant_ratio = consonant_count.div(alpha_len.replace(0, pd.NA)).fillna(0.0)
    syllables_per_char = syllable_count.div(alpha_len.replace(0, pd.NA)).fillna(0.0)
    syllables_per_token = syllable_count.div(token_count.replace(0, pd.NA)).fillna(0.0)
    mean_token_len = alpha_len.div(token_count.replace(0, pd.NA)).fillna(0.0)

    # Shape / pattern
    has_digit = no_space.str.contains("[0-9]").astype(int)
    has_hyphen = s.str.contains("-").astype(int)
    has_slash = s.str.contains("/").astype(int)
    has_parenthesis = s.str.contains(r"[()]").astype(int)
    digit_count = digits.str.len()
    punct_count = no_space.str.replace("[A-Za-z0-9]", "", regex=True).str.len()
    rare_letter_count = alpha.str.count("[xzq]")
    rare_letter_ratio = rare_letter_count.div(alpha_len.replace(0, pd.NA)).fillna(0.0)

    # Pharmacologic lexical features (simple curated set)
    stem_mab = s.str.endswith("mab").astype(int)
    stem_cillin = s.str.endswith("cillin").astype(int)
    stem_pril = s.str.endswith("pril").astype(int)
    stem_olol = s.str.endswith("olol").astype(int)
    stem_azole = s.str.endswith("azole").astype(int)

    pref_hyd = s.str.startswith("hyd").astype(int)
    pref_met = s.str.startswith("met").astype(int)

    # Formulation/context flags (token-based)
    has_hcl = s.str.contains(r"\bhcl\b").astype(int)
    has_na = s.str.contains(r"\b(na|sodium)\b").astype(int)
    has_sr = s.str.contains(r"\bsr\b").astype(int)
    has_xr = s.str.contains(r"\bxr\b").astype(int)
    has_er = s.str.contains(r"\ber\b").astype(int)

    data: Dict[str, pd.Series] = {
        f"{prefix}_alpha_len": alpha_len,
        f"{prefix}_char_len": char_len,
        f"{prefix}_syllable_count": syllable_count,
        f"{prefix}_consonant_count": consonant_count,
        f"{prefix}_vowel_count": vowel_count,
        f"{prefix}_vowel_ratio": vowel_ratio,
        f"{prefix}_consonant_ratio": consonant_ratio,
        f"{prefix}_syllables_per_char": syllables_per_char,
        f"{prefix}_token_count": token_count,
        f"{prefix}_syllables_per_token": syllables_per_token,
        f"{prefix}_mean_token_len": mean_token_len,
        f"{prefix}_has_digit": has_digit,
        f"{prefix}_has_hyphen": has_hyphen,
        f"{prefix}_has_slash": has_slash,
        f"{prefix}_has_parenthesis": has_parenthesis,
        f"{prefix}_digit_count": digit_count,
        f"{prefix}_punct_count": punct_count,
        f"{prefix}_max_token_len": max_token_len,
        f"{prefix}_rare_letter_count": rare_letter_count,
        f"{prefix}_rare_letter_ratio": rare_letter_ratio,
        f"{prefix}_stem_mab": stem_mab,
        f"{prefix}_stem_cillin": stem_cillin,
        f"{prefix}_stem_pril": stem_pril,
        f"{prefix}_stem_olol": stem_olol,
        f"{prefix}_stem_azole": stem_azole,
        f"{prefix}_pref_hyd": pref_hyd,
        f"{prefix}_pref_met": pref_met,
        f"{prefix}_has_hcl": has_hcl,
        f"{prefix}_has_na": has_na,
        f"{prefix}_has_sr": has_sr,
        f"{prefix}_has_xr": has_xr,
        f"{prefix}_has_er": has_er,
        f"{prefix}_frac_alpha_tokens": frac_alpha_tokens,
        f"{prefix}_frac_alnum_tokens": frac_alnum_tokens,
        f"{prefix}_max_token_digit_count": max_token_digit_count,
    }

    base_df = pd.DataFrame(data).astype("float32")

    # Positional/trend consonant-vowel features on the alphabetic-only
    # projection of the name to focus on linguistic structure.
    positional_df = _compute_positional_cv_features(alpha, prefix=f"{prefix}_pos")
    entropy_df = _compute_entropy_and_run_features(alpha, prefix=f"{prefix}_alpha")

    return pd.concat([base_df, positional_df, entropy_df], axis=1)


# ---------------------------------------------------------------------------
# Generic code-string helpers (ICD/CPT)
# ---------------------------------------------------------------------------


def _encode_generic_code_series(code_series: pd.Series, prefix: str) -> pd.DataFrame:
    """Shared numeric patterns for ICD/CPT-like codes."""
    s = _safe_str_series(code_series).str.strip().str.upper()
    no_space = s.str.replace(r"\s+", "", regex=True)

    char_len = no_space.str.len()
    alpha = no_space.str.replace("[^A-Z]", "", regex=True)
    digits = no_space.str.replace("[^0-9]", "", regex=True)

    alpha_count = alpha.str.len()
    digit_count = digits.str.len()

    alpha_ratio = alpha_count.div(char_len.replace(0, pd.NA)).fillna(0.0)
    digit_ratio = digit_count.div(char_len.replace(0, pd.NA)).fillna(0.0)

    unique_char_count = no_space.apply(lambda x: len(set(x)) if x else 0)

    def _max_run(ch_str: str) -> int:
        if not ch_str:
            return 0
        max_run = 1
        cur = 1
        last = ch_str[0]
        for c in ch_str[1:]:
            if c == last:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                last = c
                cur = 1
        return max_run

    max_run_same_char = no_space.apply(_max_run)

    def _numeric_prefix_len(sval: str) -> int:
        n = 0
        for ch in sval:
            if ch.isdigit():
                n += 1
            else:
                break
        return n

    def _alpha_prefix_len(sval: str) -> int:
        n = 0
        for ch in sval:
            if ch.isalpha():
                n += 1
            else:
                break
        return n

    numeric_prefix_len = no_space.apply(_numeric_prefix_len)
    alpha_prefix_len = no_space.apply(_alpha_prefix_len)

    data: Dict[str, pd.Series] = {
        f"{prefix}_char_len": char_len,
        f"{prefix}_alpha_count": alpha_count,
        f"{prefix}_digit_count": digit_count,
        f"{prefix}_alpha_ratio": alpha_ratio,
        f"{prefix}_digit_ratio": digit_ratio,
        f"{prefix}_unique_char_count": unique_char_count,
        f"{prefix}_max_run_same_char": max_run_same_char,
        f"{prefix}_numeric_prefix_len": numeric_prefix_len,
        f"{prefix}_alpha_prefix_len": alpha_prefix_len,
    }
    return pd.DataFrame(data).astype("float32")


# ---------------------------------------------------------------------------
# ICD encoding
# ---------------------------------------------------------------------------


def encode_icd_series(icd_series: pd.Series, prefix: str = "icd") -> pd.DataFrame:
    """
    Encode ICD-10 codes to numeric features based on structure and composition.
    """
    # Remove decimal but otherwise keep code structure
    s = _safe_str_series(icd_series).str.strip().str.upper().str.replace(".", "", regex=False)
    base = _encode_generic_code_series(s, prefix=prefix)

    # Positional/trend consonant-vowel features over the full cleaned code,
    # plus entropy / run-based features.
    pos_df = _compute_positional_cv_features(s, prefix=f"{prefix}_pos")
    entropy_df = _compute_entropy_and_run_features(
        s.str.replace(r"\s+", "", regex=True), prefix=f"{prefix}_code"
    )

    code_len = base[f"{prefix}_char_len"]
    has_7th_char = (code_len == 7).astype("float32")
    missing_char_count = (7 - code_len).clip(lower=0)

    c1_is_letter = s.str[0].fillna("").str.match("[A-Z]").astype("float32")
    c2_is_digit = s.str[1].fillna("").str.match("[0-9]").astype("float32")

    def _is_alnum_at(pos: int):
        return s.str[pos].fillna("").str.match("[A-Z0-9]").astype("float32")

    c3_is_alnum = _is_alnum_at(2)
    c4_is_alnum = _is_alnum_at(3)
    c5_is_alnum = _is_alnum_at(4)
    c6_is_alnum = _is_alnum_at(5)
    c7_is_alnum = _is_alnum_at(6)

    has_X_placeholder = s.str.contains("X").astype("float32")
    ends_with_A_D_S = s.str[-1].fillna("").isin(["A", "D", "S"]).astype("float32")

    # First 3 characters as "category" (may be shorter for malformed codes)
    category3 = s.str.slice(0, 3)
    # Factorize to dense integer IDs; treat missing as -1
    cat_codes, _ = pd.factorize(category3, sort=False)
    category_idx = pd.Series(cat_codes).astype("float32")

    # Chapter index from first character bucket (very coarse)
    first_char = s.str.slice(0, 1)

    def _chapter_idx(ch: str) -> int:
        if not ch:
            return -1
        o = ord(ch)
        # Simple buckets A-B=1, C-D=2, E-H=3, I-K=4, L-M=5, N-R=6, S-T=7, V-Y=8, others=9
        if "A" <= ch <= "B":
            return 1
        if "C" <= ch <= "D":
            return 2
        if "E" <= ch <= "H":
            return 3
        if "I" <= ch <= "K":
            return 4
        if "L" <= ch <= "M":
            return 5
        if "N" <= ch <= "R":
            return 6
        if "S" <= ch <= "T":
            return 7
        if "V" <= ch <= "Y":
            return 8
        return 9

    chapter_idx = first_char.apply(_chapter_idx).astype("float32")

    # Suffix lengths
    def _suffix_lens(code: str) -> Tuple[int, int]:
        if not code:
            return 0, 0
        num = 0
        alpha = 0
        # numeric suffix
        for ch in reversed(code):
            if ch.isdigit():
                num += 1
            else:
                break
        # alpha suffix
        for ch in reversed(code):
            if ch.isalpha():
                alpha += 1
            else:
                break
        return num, alpha

    suffix_vals = s.apply(_suffix_lens)
    numeric_suffix_len = suffix_vals.apply(lambda t: t[0])
    alpha_suffix_len = suffix_vals.apply(lambda t: t[1])

    extra = pd.DataFrame(
        {
            f"{prefix}_has_7th_char": has_7th_char,
            f"{prefix}_missing_char_count": missing_char_count,
            f"{prefix}_c1_is_letter": c1_is_letter,
            f"{prefix}_c2_is_digit": c2_is_digit,
            f"{prefix}_c3_is_alnum": c3_is_alnum,
            f"{prefix}_c4_is_alnum": c4_is_alnum,
            f"{prefix}_c5_is_alnum": c5_is_alnum,
            f"{prefix}_c6_is_alnum": c6_is_alnum,
            f"{prefix}_c7_is_alnum": c7_is_alnum,
            f"{prefix}_has_X_placeholder": has_X_placeholder,
            f"{prefix}_ends_with_A_D_S": ends_with_A_D_S,
            f"{prefix}_category_idx": category_idx,
            f"{prefix}_chapter_idx": chapter_idx,
            f"{prefix}_numeric_suffix_len": numeric_suffix_len,
            f"{prefix}_alpha_suffix_len": alpha_suffix_len,
        }
    ).astype("float32")

    return pd.concat([base, pos_df, entropy_df, extra], axis=1)


# ---------------------------------------------------------------------------
# CPT encoding
# ---------------------------------------------------------------------------


def encode_cpt_series(cpt_series: pd.Series, prefix: str = "cpt") -> pd.DataFrame:
    """
    Encode CPT codes (5-char codes) into numeric features.
    """
    s = _safe_str_series(cpt_series).str.strip().str.upper()
    base = _encode_generic_code_series(s, prefix=prefix)

    # Positional/trend consonant-vowel features over the full cleaned code,
    # plus entropy / run-based features.
    pos_df = _compute_positional_cv_features(s, prefix=f"{prefix}_pos")
    entropy_df = _compute_entropy_and_run_features(
        s.str.replace(r"\s+", "", regex=True), prefix=f"{prefix}_code"
    )

    code_len = base[f"{prefix}_char_len"]

    # First 3 and last 2 as integers where numeric
    first3 = s.str.slice(0, 3)
    last2 = s.str.slice(-2, None)

    def _as_int(text: str) -> int:
        if text and text.isdigit():
            return int(text)
        return 0

    first3_int = first3.apply(_as_int)
    last2_int = last2.apply(_as_int)

    all_digits = s.str.fullmatch(r"\d{5}").fillna(False).astype("float32")
    ends_with_F = s.str.endswith("F").astype("float32")
    ends_with_T = s.str.endswith("T").astype("float32")
    has_letter = s.str.contains("[A-Z]").astype("float32")

    first_char = s.str.slice(0, 1)

    def _first_digit_int(ch: str) -> int:
        if ch.isdigit():
            return int(ch)
        if not ch:
            return -1
        # encode letters as 9
        return 9

    first_digit_int = first_char.apply(_first_digit_int)

    # Prefix buckets
    prefix_bucket = first3.apply(
        lambda x: int(x[0]) if x and x[0].isdigit() else -1
    ).astype("float32")

    # Coarse groupings
    hundreds_bin = first3_int // 100
    tens_bin = first3_int // 10

    extra = pd.DataFrame(
        {
            f"{prefix}_first3_int": first3_int.astype("float32"),
            f"{prefix}_last2_int": last2_int.astype("float32"),
            f"{prefix}_all_digits": all_digits,
            f"{prefix}_ends_with_F": ends_with_F,
            f"{prefix}_ends_with_T": ends_with_T,
            f"{prefix}_has_letter": has_letter,
            f"{prefix}_first_digit_int": first_digit_int.astype("float32"),
            f"{prefix}_prefix_bucket": prefix_bucket,
            f"{prefix}_hundreds_bin": hundreds_bin.astype("float32"),
            f"{prefix}_tens_bin": tens_bin.astype("float32"),
        }
    )

    return pd.concat([base, pos_df, entropy_df, extra], axis=1).astype("float32")


