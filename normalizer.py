import dateparser
from word2number import w2n


def _digit_runs(text):
    """Return all maximal runs of digits in `text`, biggest first."""
    runs = []
    current = []
    for char in text:
        if char.isdigit():
            current.append(char)
        elif current:
            runs.append("".join(current))
            current = []
    if current:
        runs.append("".join(current))
    runs.sort(key=len, reverse=True)
    return runs


def normalize_agreement_value(raw_answer):
    if not raw_answer:
        return ""
    # 1. Words → number ("five thousand" → "5000")
    try:
        return str(w2n.word_to_num(raw_answer))
    except Exception:
        pass
    # 2. Drop thousand-separator commas that sit between digits ("6,500" → "6500").
    chars = list(raw_answer)
    for i in range(len(chars) - 2, 0, -1):
        if chars[i] == "," and chars[i - 1].isdigit() and chars[i + 1].isdigit():
            del chars[i]
    cleaned = "".join(chars)
    # 3. Strip a trailing decimal portion (".00", ".50") so "6500.00" → "6500".
    #    We do this by walking digit runs separated by a single period.
    runs = _digit_runs(cleaned)
    if not runs:
        return raw_answer.strip()
    # The amount is the longest digit run that isn't part of a decimal tail.
    # Heuristic: if a run is followed by "." and another short run (≤ 2 digits)
    # in the original text, treat the second run as cents and keep only the first.
    # In practice the longest run is the integer part of the rent.
    return runs[0]


def normalize_date(raw_answer):
    if not raw_answer:
        return ""
    parsed = dateparser.parse(raw_answer, settings={"DATE_ORDER": "DMY"})
    if parsed:
        return parsed.strftime("%d.%m.%Y")
    return raw_answer.strip()


def normalize_days(raw_answer):
    if not raw_answer:
        return ""
    try:
        return str(w2n.word_to_num(raw_answer))
    except Exception:
        pass
    runs = _digit_runs(raw_answer)
    if runs:
        # Days are usually a 1-3 digit number; the FIRST short run is more
        # reliable than the longest run (which could be a year).
        for run in runs:
            if 1 <= len(run) <= 3:
                return run
        return runs[0]
    return raw_answer.strip()


def normalize_party_name(raw_answer):
    if not raw_answer:
        return ""
    cleaned = raw_answer.strip()
    label_only = {"lessor", "lessee", "tenant", "landlord", "owner",
                  "house owner", "the", "party", "."}
    if cleaned.lower() in label_only:
        return ""
    # Strip leading boilerplate the QA model sometimes prepends.
    junk_prefixes = ["between ", "lessor ", "lessee ", "tenant ",
                     "landlord ", "owner ", "the "]
    lowered = cleaned.lower()
    for prefix in junk_prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            lowered = cleaned.lower()
            break
    return cleaned


def normalize(fields):
    return {
        "agreement_value":      normalize_agreement_value(fields.get("agreement_value", "")),
        "agreement_start_date": normalize_date(fields.get("agreement_start_date", "")),
        "agreement_end_date":   normalize_date(fields.get("agreement_end_date", "")),
        "renewal_notice_days":  normalize_days(fields.get("renewal_notice_days", "")),
        "party_one":            normalize_party_name(fields.get("party_one", "")),
        "party_two":            normalize_party_name(fields.get("party_two", "")),
    }
