## Feature Encoding Strategy for Drug Names

This document describes a feature encoding strategy that converts raw drug name strings into a compact, numeric representation suitable for machine learning models. The focus is on interpretable, low-dimensional features that capture length, phonetic structure, and orthographic patterns of drug names.

### Overview

The goal is to derive features that:

- Preserve clinically meaningful patterns in drug nomenclature.
- Are numeric and model-friendly (CatBoost, XGBoost, etc.).
- Are stable across modeling frameworks (do not depend on internal CTR/hash maps).

This is useful for:

- Drug name classification and grouping.
- Error detection and string-matching tasks.
- Providing explainable, model-agnostic encodings that can be shared across models and explainers.

### Preprocessing

Before feature extraction, normalize each drug name:

- Convert to lowercase.
- Strip leading/trailing whitespace.
- Optionally normalize punctuation (e.g., unify hyphens, slashes, parentheses) depending on downstream use.
- Optionally keep a parallel "raw" version if case or punctuation patterns are informative.

Let the normalized string be denoted as `name`, and its whitespace-split tokens as `tokens`.

### Core Length and Phonetic Features

For each `name`, compute:

- `alpha_len`: Number of alphabetic characters (`a`–`z`).
- `char_len`: Total number of characters in the normalized string, excluding spaces (letters, digits, punctuation).
- `syllable_count`: Total number of syllables in the drug name, computed by summing syllables over tokens using a syllable library (e.g., Pyphen) or a heuristic function.
- `consonant_count`: Number of consonant letters (alphabetic characters not in a vowel set such as `aeiouy`).
- `vowel_count`: Number of vowels (letters in the chosen vowel set).

These implement:

1. Length of drug (string) -> `char_len` and `alpha_len`.
2. Number of syllables in drug (string) -> `syllable_count`.
3. Number of consonants in drug (string) -> `consonant_count`.
4. Number of characters -> `char_len`.

### Composition and Ratio Features

Ratios normalize for name length and help models generalize across short and long names:

- `vowel_ratio` = `vowel_count` / `alpha_len` (0 if `alpha_len == 0`).
- `consonant_ratio` = `consonant_count` / `alpha_len` (0 if `alpha_len == 0`).
- `syllables_per_char` = `syllable_count` / `alpha_len` (0 if `alpha_len == 0`).
- `token_count` = number of whitespace-separated tokens.
- `syllables_per_token` = `syllable_count` / `token_count` (0 if `token_count == 0`).
- `mean_token_len` = `alpha_len` / `token_count` (0 if `token_count == 0`).

### Shape and Pattern Features

Binary indicator (0/1) features:

- `has_digit`: At least one numeric character.
- `has_hyphen`: At least one `-`.
- `has_slash`: At least one `/`.
- `has_parenthesis`: At least one `(` or `)`.

Count features:

- `digit_count`: Number of digit characters.
- `punct_count`: Number of punctuation/special characters (excluding letters, digits, and spaces).
- `max_token_len`: Maximum alphabetic length across tokens.
- `rare_letter_count`: Number of occurrences of rare letters such as `x`, `z`, `q`.
- `rare_letter_ratio` = `rare_letter_count` / `alpha_len` (0 if `alpha_len == 0`).

### Pharmacologically Motivated Lexical Features

Drug names often follow International Nonproprietary Name (INN) conventions and contain characteristic stems and affixes.

Binary (0/1) stem flags from curated suffixes/prefixes (examples):

- Suffix flags:
  - `stem_mab` (monoclonal antibodies): 1 if `name` ends with `mab`.
  - `stem_cillin`: ends with `cillin`.
  - `stem_pril`: ends with `pril`.
  - `stem_olol`: ends with `olol`.
  - `stem_azole`: ends with `azole`.

- Prefix flags (examples):
  - `pref_hyd`: starts with `hyd`.
  - `pref_met`: starts with `met`.

Binary "formulation/context" flags (examples):

- `has_hcl`: token `hcl` present.
- `has_na`: token `na` or `sodium` present.
- `has_sr`: token `sr` present.
- `has_xr`: token `xr` present.
- `has_er`: token `er` present.

Token-shape features:

- `frac_alpha_tokens`: fraction of tokens that are purely alphabetic.
- `frac_alnum_tokens`: fraction of tokens containing both letters and digits.
- `max_token_digit_count`: maximum number of digits in any token.

### Recommended Feature Set Summary

For a practical, compact numeric encoding, the recommended features are:

- Length/phonetic:
  - `alpha_len`, `char_len`, `syllable_count`, `consonant_count`, `vowel_count`.
- Ratios:
  - `vowel_ratio`, `consonant_ratio`, `syllables_per_char`, `token_count`, `syllables_per_token`, `mean_token_len`.
- Shape/pattern:
  - `has_digit`, `has_hyphen`, `has_slash`, `has_parenthesis`, `digit_count`, `punct_count`, `max_token_len`, `rare_letter_count`, `rare_letter_ratio`.
- Pharmacologic lexical:
  - A curated set of 10–20 stem flags (e.g., `stem_mab`, `stem_cillin`, `stem_pril`, `stem_olol`, `stem_azole`, selected prefixes).
  - Formulation/context flags (e.g., `has_hcl`, `has_na`, `has_sr`, `has_xr`, `has_er`).
  - Token-shape fractions (`frac_alpha_tokens`, `frac_alnum_tokens`, `max_token_digit_count`).

---

## Positional and Trend-Based Consonant/Vowel Features

On top of the string-level encodings above, we add a compact set of **positional** and **trend-based** features that summarize how vowels and consonants are distributed across a string (drug name, ICD code, CPT code).

### Positional setup

For each cleaned string:

- Let \(L\) be the character length with indices \(i = 0, \dots, L-1\).
- Define normalized positions: \(\text{pos\_norm}[i] = i / (L - 1)\) (or 0 if \(L = 1\)).
- Define vowel/consonant indicators:
  - `cv[i] = 1` for vowels, `0` for consonants or non-letters.
  - `cv_signed[i] = +1` for vowels, `-1` for consonants, `0` for non-letters.

We do **not** store these arrays directly; instead, we summarize them into a handful of scalar features.

### Trend features

- `*_cv_slope`: Slope from a simple linear regression of `cv_signed` on `pos_norm`.
  - Positive → vowels more common toward the end of the string.
  - Negative → vowels more common near the beginning.
- `*_vowel_ratio_first_half`: Vowel ratio among characters in the first half of the string.
- `*_vowel_ratio_second_half`: Vowel ratio among characters in the second half.
- `*_vowel_ratio_delta` = `second_half - first_half` vowel ratio.

These provide a robust, interpretable summary of how vowel/consonant composition changes from left to right.

### Positional concentration features

- `*_center_of_vowels`: Mean `pos_norm` across vowel positions (defaults to 0.5 if no vowels).
- `*_center_of_consonants`: Mean `pos_norm` across consonant positions (defaults to 0.5 if no consonants).
- `*_center_diff` = `center_of_vowels - center_of_consonants`.

These single scalars indicate whether vowels are skewed earlier or later than consonants.

### Binary pattern indicators

- `*_starts_with_vowel`: 1 if the first character is a vowel.
- `*_ends_with_vowel`: 1 if the last character is a vowel.
- `*_vowel_majority_first_half`: 1 if vowels outnumber consonants in the first half.
- `*_vowel_majority_second_half`: 1 if vowels outnumber consonants in the second half.
- `*_has_vowel_run_end`: 1 if the last N characters (we use N=3) are all vowels.

These flags are compact, model-friendly, and show up clearly in feature importance and FFA explanations.

---

## Additional Advanced Numeric String Features

To further separate codes and names numerically while keeping the representation compact and interpretable, we add a small set of **entropy and run-based features**. These are computed for drug names and medical codes (ICD, CPT) and then aggregated to the patient level (mean / max) in the final model builder.

### Entropy and diversity

For each cleaned string (e.g., alphabetic drug name or code without spaces):

- `*_char_entropy`: Shannon entropy of the character distribution.
  - High values → many distinct characters with similar frequencies.
  - Low values → repetitive or highly structured strings.
- `*_bigram_diversity`: `unique_bigrams / (L - 1)` where bigrams are contiguous 2-character substrings and \(L\) is the string length (0 if \(L < 2\)).

These capture how “varied” the character-level pattern is beyond simple length and counts.

### Consonant-cluster structure and transitions

On the alphabetic-only projection of the string:

- `*_max_consonant_run_len`: Longest run of consecutive consonants.
- `*_mean_consonant_run_len`: Mean length across all consonant runs.
- `*_vc_transition_rate`: Number of vowel↔consonant transitions divided by the number of transitions (`alpha_len - 1`, with safe handling when `alpha_len <= 1`).
- `*_tri_consonant_cluster_count`: Count of windows of length 3 where all three characters are consonants.
- `*_vowel_run_count_ge2`: Number of vowel runs with length ≥ 2.

These metrics provide a more nuanced description of how “chunky” or alternating the consonant/vowel pattern is, which is particularly useful for distinguishing stems, affixes, and code patterns without resorting to one-hot encodings.

---

## ICD Code Feature Encoding

ICD‑10‑CM/PCS codes are alphanumeric, typically 3–7 characters long, with character‑level meaning (1st char letter, 2nd numeric, others alphanumeric, decimal after 3rd).

### Preprocessing (per code)

- Normalize to uppercase.
- Remove the decimal point (keep a parallel raw version if needed).
- Let `code` be the cleaned string, `L = len(code)`.

### Numeric Features

- **Length / position**
  - `code_len`: number of characters (3–7 for ICD‑10‑CM, 7 for ICD‑10‑PCS).
  - `has_7th_char`: 1 if `code_len == 7`, else 0.
  - `missing_char_count`: `max(0, 7 - code_len)` (for ICD‑10‑CM).

- **Character class counts**
  - `alpha_count`: count of letters.
  - `digit_count`: count of digits.
  - `alpha_ratio` = `alpha_count / code_len` (0 if `code_len == 0`).
  - `digit_ratio` = `digit_count / code_len` (0 if `code_len == 0`).

- **Position‑specific encodings**
  - `c1_is_letter`: 1 if first char is `A`–`Z` (QA check for valid ICD‑10).
  - `c2_is_digit`: 1 if second char is `0`–`9`.
  - `c3_is_alnum`, `c4_is_alnum`, …, `c7_is_alnum`: 1 if position exists and is alphanumeric.
  - Optional low‑cardinality mappings:
    - `chapter_idx`: integer bucket derived from `code[0]` (e.g., A–B=1, C–D=2, etc.).
    - `category_idx`: dense integer ID for `code[0:3]` (e.g., via frequency rank).

- **Pattern / shape flags**
  - `has_X_placeholder`: 1 if `X` appears as a placeholder character.
  - `ends_with_A_D_S`: 1 if last char in `{A, D, S}` (common injury/encounter extensions).
  - `numeric_suffix_len`: length of trailing digit run.
  - `alpha_suffix_len`: length of trailing letter run.

These features give a compact numeric summary of ICD structure without exploding into one‑hot vectors per code.

---

## CPT Code Feature Encoding

CPT codes are five characters; Category I are mostly 5 digits, while Category II and III include terminal letters (e.g., `0123T`).

### Preprocessing (per code)

- Normalize to uppercase.
- Strip spaces or modifiers.
- If modifiers are present (e.g., `-26`, `-TC`), parse them into **separate fields** for modifier features.

### Numeric Features

- **Base structure**
  - `code_len`: number of characters (typically 5).
  - `alpha_count`, `digit_count`, `alpha_ratio`, `digit_ratio` as defined above.
  - `first3_int`: integer value of first 3 characters if all digits, else 0.
  - `last2_int`: integer value of last 2 characters if both digits, else 0.

- **Category and pattern flags**
  - `all_digits`: 1 if all 5 chars are digits (tends to be Category I).
  - `ends_with_F`: 1 if last char is `F` (Category II).
  - `ends_with_T`: 1 if last char is `T` (Category III).
  - `has_letter`: 1 if any letter present.
  - `prefix_0x`, `prefix_1x`, …: indicator on first digit bucket (e.g., first char `0`, `1`, `2`, etc.).
  - `first_digit_int`: numeric value of first char if digit, else special code (e.g., 9).

- **Grouping encodings**
  - `hundreds_bin` = `floor(first3_int / 100)` (0–99, 100–199, …).
  - `tens_bin` = `floor(first3_int / 10)` for narrower groupings.

- **Modifier‑style features** (if parsed separately)
  - Binary flags per common modifier: `mod_26`, `mod_TC`, `mod_59`, etc.

These features exploit CPT’s fixed length and simple numeric/alpha pattern without creating high‑cardinality one‑hots.

---

## Shared ICD/CPT Numeric Patterns

A small shared block of generic code‑string features can be reused for both ICD and CPT (and other code systems):

- `char_len`: number of characters in the cleaned code.
- `alpha_count`, `digit_count`, `alpha_ratio`, `digit_ratio`.
- `unique_char_count`: number of distinct characters used.
- `max_run_same_char`: longest run of identical characters (e.g., `"000"`).
- `numeric_prefix_len`: length of leading digit run.
- `alpha_prefix_len`: length of leading letter run.

These are purely numeric, easy to compute, and provide regularizing signal for models working with mixed code systems.

### Rationale for This Encoding

- **Model-agnostic**: All features are simple numeric columns; they can be used by CatBoost, XGBoost, logistic regression, and any explainer that expects a dense numeric matrix.
- **Explainable**: Each feature has a clear linguistic or pharmacologic interpretation (e.g., "drug ends with mab", "high rare-letter ratio").
- **Stable across frameworks**: Because encoding is done **upfront in the feature table**, we avoid framework-specific CTR/hash structures (e.g., CatBoost's internal categorical encodings), which simplifies FFA and causal analysis.

References and inspiration:

- Traditional text feature engineering approaches (character and syllable level).
- INN stem lists and common drug naming conventions.

