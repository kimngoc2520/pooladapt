"""Inference-only frozen Phase 2 feature and PACE-EF input construction."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.baselines.pace_ef import bm25_document_contribution, evidence_frontload, normalize_rrf_scores
from src.pooladapt.selector import FROZEN_FEATURES
from src.retrieval.bm25 import BM25Retriever, tokenize


def validate_pool(pool: Sequence[Mapping[str, Any]]) -> None:
    """Require exactly 100 unique documents, without relevance fields."""
    if len(pool) != 100 or len({str(c['id']) for c in pool}) != 100:
        raise ValueError("expected exactly 100 unique candidate documents")
    if any(any(token in key.lower() for token in ('qrel', 'relevance', 'm_star', 'label')) for c in pool for key in c):
        raise ValueError("relevance fields must never enter selector inputs")


def frozen_features(query: str, pool: Sequence[Mapping[str, Any]],
                    sparse: Sequence[Mapping[str, Any]], dense: Sequence[Mapping[str, Any]],
                    corpus: Mapping[str, Mapping[str, Any]], bm25: BM25Retriever) -> dict[str, float]:
    """Reproduce run_candidate_logging's 13 features without labels or IDs."""
    validate_pool(pool)
    bm, de = {c['id']: c for c in sparse}, {c['id']: c for c in dense}
    scores = np.asarray([c['rrf_score'] for c in pool], dtype=float)
    probs = scores / scores.sum()
    paired = np.asarray([(bm[x]['rank'], de[x]['rank']) for x in set(bm) | set(de) if x in bm and x in de])
    rho = float(np.corrcoef(paired.T)[0, 1]) if len(paired) > 1 and np.std(paired, axis=0).min() > 0 else np.nan
    gaps = [abs(bm[c['id']]['rank'] - de[c['id']]['rank']) for c in pool if c['id'] in bm and c['id'] in de]
    disagreement = float(np.mean(gaps) / 99) if gaps else np.nan
    token_sets = [set(tokenize(corpus[c['id']].get('title', '') + ' ' + corpus[c['id']].get('text', ''))) for c in pool[:20]]
    jac = [len(a & b) / len(a | b) for i, a in enumerate(token_sets) for b in token_sets[:i] if a | b]
    sorted_scores = np.sort(scores)
    gini = float(2 * np.dot(np.arange(1, 101), sorted_scores) / (100 * sorted_scores.sum()) - 101 / 100)
    tokens = tokenize(query)
    idfs = [bm25._index.idf[t] for t in tokens if t in bm25._index.idf]
    values = (scores[0] - scores[1], len({c['id'] for c in sparse[:20]} & {c['id'] for c in dense[:20]}) / 20,
              rho, float(-(probs * np.log(probs)).sum()), gini, float(np.mean(jac)) if jac else np.nan,
              float(scores[:10].sum() / scores.sum()), 1 - disagreement, disagreement,
              len(tokens), sum(t.isnumeric() for t in tokens), float(np.mean(idfs)) if idfs else np.nan,
              max(idfs) if idfs else np.nan)
    return dict(zip(FROZEN_FEATURES, values))


def pace_order(query: str, pool: Sequence[Mapping[str, Any]], bm25: BM25Retriever) -> list[dict[str, Any]]:
    """Apply the existing BM25 adaptation of EF, preserving the entire pool."""
    validate_pool(pool)
    weights = {t: float(bm25._index.idf[t]) for t in dict.fromkeys(tokenize(query)) if t in bm25._index.idf}
    rhos = normalize_rrf_scores([float(c['rrf_score']) for c in pool])
    candidates = []
    for rank, (candidate, rho) in enumerate(zip(pool, rhos), 1):
        tokens = tokenize(str(candidate.get('title', '')) + ' ' + str(candidate.get('text', '')))
        counts = Counter(tokens)
        factors = {t: bm25_document_contribution(counts[t], len(tokens), bm25._index.k1, bm25._index.b, bm25._index.avgdl) for t in weights}
        candidates.append(dict(candidate, rho=rho, original_rank=rank, term_contributions=factors))
    ordered = evidence_frontload(candidates, weights)
    validate_pool(ordered)
    if {c['id'] for c in ordered} != {c['id'] for c in pool}:
        raise ValueError("Evidence Frontloading changed the candidate set")
    return ordered
