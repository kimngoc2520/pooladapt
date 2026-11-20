# CT2 Historical Protocol Recovery and Consolidation Record

**Date:** 2026-11-20

**Purpose:** Record the exact historical protocol content referenced by the CT2 pre-execution manifests and preserve its provenance while removing the obsolete working-tree copy.

**Scope:** Documentation and provenance only. No experiment, result, manifest, or canonical protocol content is changed by this record.

## Recovered historical content

- **Expected file-content SHA-256:** `b683345989a066f30c28ac774512282891e78fc7855156a43520a487866216e1`
- **Matching Git blob:** `c913f8007129b3c9c833344a14aba132646c1763`
- **Historical commit:** `3c90f224c52b8927222339b2f7df888b5655e706` (`Add adaptive reranking controller, feature ablation, and budget allocation analysis`)
- **Path in that commit:** `improvement/improvement_2/docs/CT2_Experiment_Protocol_draft_20261118.md`
- **Blob size:** 24,878 bytes

The SHA-256 was computed over the exact Git blob bytes and matched the expected full digest. The object is reachable in repository history; this is an exact content match, not a similarity-based reconstruction.

## Manifest references

All three manifests identify the same path, expected digest, and `DRAFT_NOT_FROZEN` status. Each records repository commit `2977ebfb2a4eaa18945c129bd0f4448c30f4fddc`.

1. `improvement/improvement_2/audit/pre_execution/gate_a_pending/input_manifest.json` — run `ct2_0_gate_a_20261118_pending`; status `PINNED_FOR_REVIEW_NOT_YET_STAGE_VALIDATED`.
2. `improvement/improvement_2/audit/pre_execution/ablation_pending/input_manifest.json` — run `ct2_1_ablation_20261118_pending`; status `PINNED_FOR_REVIEW_NOT_YET_STAGE_VALIDATED`.
3. `improvement/improvement_2/audit/pre_execution/controller_diagnosis_pending/input_manifest.json` — run `ct2_2_controller_diagnosis_20261118_pending`; status `PARTIALLY_PINNED_UPSTREAM_OOF_MISSING`.

Commit `2977ebfb2a4eaa18945c129bd0f4448c30f4fddc` predates the commit containing this protocol path, and the path is absent from its tree. The matching bytes were subsequently committed at `3c90f224c52b8927222339b2f7df888b5655e706`. Therefore the recovered digest establishes **content identity** for the manifests' protocol reference; it does not establish that the later commit existed when the manifest's repository commit was recorded. The historical manifests remain unchanged.

## Working-tree line-ending note

On this checkout, `core.autocrlf=true`. The draft's current Windows working-tree bytes measured 25,073 bytes and SHA-256 `462818194D18563B56A6CEEB679E7BA0645D86AF3822B85DBBA329F6F3AF5E1C`. Git normalizes the file to the 24,878-byte blob above; `git hash-object` maps the working-tree file to blob `c913f8007129b3c9c833344a14aba132646c1763`. Thus the working-tree byte digest differs from the expected digest because of line-ending normalization, while the normalized Git blob's exact bytes match `b6833459…`.

## Current governing protocol and approval scope

`improvement/improvement_2/audit/CT2_Experiment_Protocol.md` remains the intended current governing protocol document. Its current working-tree SHA-256 at the time of this record is `47E612C0B85FA40ADAF1CD94AA6781B3FF215A22DE79C203A4E0421726E6B146`.

The audit protocol remains an overall **draft / not frozen**. Amendment A1 records approval of the current schema only: Full=12, Pool-only=8, Query-only=4; `sparse_dense_disagreement` is retained and `sparse_dense_agreement` is excluded. Historical 13-feature artifacts remain legacy. A1 does not approve other pending methods or Gate E criteria; those retain their stated pending status.

## Separate protocol-hash history caveat

This recovery does not reconcile protocol-version hashes. The current audit protocol working-tree digest recorded above is `47E612C0…`. A separate prior reference supplied as `510F…` has no complete digest or version mapping in this recovery task. These values must not be silently substituted for one another or for the recovered historical draft digest. Reconcile their protocol versions, byte/line-ending conventions, and run mappings in a separate history audit if needed.

Future execution instructions should identify the audit protocol path when the applicable protocol version has been approved/frozen and should record that exact version/hash in the run manifest. This note does not freeze or approve the current draft.
