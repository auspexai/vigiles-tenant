"""Vigiles drift-probe executor — the D6 work unit (minimal, deterministic).

One unit = one fixed probe (a scripted message sequence) generated against
the worker-served local model through the AuspexAI inference broker, reduced
to light, deterministic lexical features. The features — not the raw text —
are the result payload:

- AuspexAI's hash-agreement consensus needs byte-identical replica payloads;
  every field here is deterministic given deterministic generation (the
  broker pins temperature 0 + the seed; the model+quant is pinned by the
  manifest's exact model id).
- Per the Research Ethics Policy §7 containment posture, raw model output
  is research evidence, not a deliverable — the payload carries the
  response's sha256 and aggregate lexical features only. Drift analysis
  compares those features (and hash stability) across rounds, tenant-side.

Runs under the worker sandbox's system Python with zero installs: the only
import is the vendored `lite` module (stdlib-only LiteHarness +
InferenceClient from the tenant SDK's vendorable kit).

Unit payload shape:

    {"probe_id": "p-greeting",
     "messages": [{"role": "user", "content": "..."}],
     "seed": 0,
     "num_predict": 256}

Result payload shape (vigiles-drift-probe/v0):

    {"schema": "vigiles-drift-probe/v0",
     "probe_id": "p-greeting",
     "response_sha256": "...",          # the drift/consensus anchor
     "response_chars": N,
     "eval_count": N,                   # backend-reported completion tokens
     "lexical": {"tokens": N, "unique_tokens": N,
                 "type_token_ratio": 0.x, "top_tokens": [["the", 4], ...]},
     "model": {"id": "<model_id>", "gguf_sha256": "..."}}  # provenance
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter

from lite import InferenceClient, LiteHarness

TOP_TOKENS = 8


def _strip_surrounding_punct(token: str) -> str:
    """Drop leading/trailing UNICODE punctuation from a token (every category
    Pc/Pd/Pe/Pf/Pi/Po/Ps — ASCII commas/periods AND the em-dashes, curly quotes,
    and ellipses LLMs emit constantly). Interior marks (the apostrophe in "i'm",
    dots in "u.s") are preserved."""
    lo, hi = 0, len(token)
    while lo < hi and unicodedata.category(token[lo])[0] == "P":
        lo += 1
    while hi > lo and unicodedata.category(token[hi - 1])[0] == "P":
        hi -= 1
    return token[lo:hi]


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenization, SURROUNDING punctuation trimmed for stability.

    Whitespace split, then strip leading/trailing punctuation from each token and
    drop empties. The trimming is load-bearing for cross-version consensus: the C7
    proof run saw "blue, red, yellow" diverge under within_cell_tolerance ONLY
    because a comma attached to a different word across Ollama versions
    (`red,`/`yellow,` vs `blue,`/`yellow`) — identical behavior must produce
    identical tokens, or a trivial tokenization artifact masquerades as drift.
    Unicode-aware so curly quotes / em-dashes don't reintroduce the same brittleness.
    Still deterministic and a pure function of the text."""
    out: list[str] = []
    for raw in text.lower().split():
        tok = _strip_surrounding_punct(raw)
        if tok:
            out.append(tok)
    return out


def run_one(unit: dict, models_dir) -> dict:
    payload = unit["payload"]
    client = InferenceClient.from_env()
    info = client.info()

    reply = client.generate(
        payload["messages"],
        options={
            "seed": int(payload.get("seed", 0)),
            "num_predict": int(payload.get("num_predict", 256)),
        },
    )
    text = reply["message"]["content"]

    tokens = _tokenize(text)
    counts = Counter(tokens)
    # Deterministic ordering: count desc, then token asc.
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_TOKENS]

    return {
        "schema": "vigiles-drift-probe/v0",
        "probe_id": payload.get("probe_id"),
        "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "response_chars": len(text),
        "eval_count": int(reply.get("eval_count", 0)),
        "lexical": {
            "tokens": len(tokens),
            "unique_tokens": len(counts),
            "type_token_ratio": round(len(counts) / len(tokens), 6) if tokens else 0.0,
            "top_tokens": [[t, c] for t, c in top],
        },
        "model": {"id": reply.get("model"), "gguf_sha256": info.get("gguf_sha256")},
    }


if __name__ == "__main__":
    import sys

    sys.exit(LiteHarness(run_one).main())
