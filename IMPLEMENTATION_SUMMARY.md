# Implementation Summary: Entity-Count Filter Removal

**Date**: January 8, 2026  
**Status**: ✅ Implemented  
**Files Modified**: 2  
**Documentation Created**: 2

---

## Changes Made

### 1. Code Changes

#### **File: `indexer.py`**

- **Lines**: 575-595 (pruneNoise method)
- **Change**: Removed entity-count filtering logic
- **Impact**: Chunks are no longer evaluated/removed based on entity count
- **Added**: Detailed code comments explaining rationale

#### **File: `graphrag_config.py`**

- **Lines**: 177-196 (Garbage Filtering Parameters section)
- **Change**: Removed `FILTER_MIN_ENTITIES: int = 1` parameter
- **Impact**: Eliminates configuration that caused over-pruning
- **Added**: Documentation explaining removal and listing remaining filters

---

### 2. Documentation Created

#### **File: `project_documentation/contextual_entity_filtering.md`**

- **Purpose**: Documents Option 2 (neighbor-aware filtering) for future consideration
- **Contents**:
  - Design specification and algorithm
  - Example scenarios
  - Implementation considerations (pros/cons)
  - Decision criteria for when to implement
  - Integration points and testing strategy

#### **File: `project_documentation/Indexing_troubleshooting_documentation.md`**

- **Update**: Added section documenting the permanent fix
- **Contents**:
  - What changed and why
  - Current pruning behavior
  - Reference to future enhancement option

---

## Rationale Summary

**Why remove entity-count filtering?**

1. **Not a quality signal**: Legitimate chunks (summaries, introductions, methodology) often have 0 entities
2. **GLiNER behavior**: Produces bimodal distribution (0 or 3+ entities), making thresholds useless
3. **Over-pruning**: Deleted 4,667/4,770 chunks (99%) in the user's case
4. **Redundant**: Existing filters (entropy, repetition, malformed) catch actual garbage
5. **Domain-specific**: Financial docs have more entities than scientific papers → not transferable
6. **KISS principle**: Simpler is better; avoid unnecessary configuration

**Remaining Filters (Still Active):**

- `FILTER_REPETITION_THRESHOLD` - Catches copy-paste junk
- `FILTER_MIN_ENTROPY` / `FILTER_MAX_ENTROPY` - Catches random noise
- `FILTER_MALFORMED_THRESHOLD` - Catches OCR errors
- `FILTER_MAX_WHITESPACE_DENSITY` - Catches formatting artifacts
- (Optional) `FILTER_QUALITY_THRESHOLD` - LLM-based quality scoring
- (Optional) `FILTER_EMBEDDING_OUTLIER_THRESHOLD` - Statistical anomaly detection

---

## Next Steps for User

### Immediate Action Required

Re-index the database to recover deleted data:

```bash
docker compose run --rm graphrag python indexer.py --reset --extraction-mode gliner_llm
```

**Expected results:**

- ~4,700-5,000 chunks (up from 103)
- Entities for all companies (IREN, RKLB, SLV, META, NVDA, etc.)
- Queries return relevant results

### Verification

After re-indexing:

```bash
# Check stats
docker compose run --rm graphrag python check_db.py .DuckDB/graphrag.duckdb

# Test RKLB query
docker compose run --rm graphrag python query_engine.py "What can we expect of RKLB in 2026?" --type explore_thematic --topk 5
```

---

## Long-Term Implications

### Pruning is Now Safe

The `--prune` flag can be used safely in future indexing runs:

```bash
docker compose run --rm graphrag python indexer.py --extraction-mode gliner_llm --prune
```

It will only remove genuinely low-quality chunks (noise, artifacts, malformed text), not contextual content.

### Future Enhancement Path

If simple filtering proves insufficient, the **contextual entity filtering** approach (Option 2) can be implemented:

- Documented in `project_documentation/contextual_entity_filtering.md`
- Adds neighbor-aware logic (keep 0-entity chunks within entity-rich sections)
- Deferred until proven necessary (YAGNI principle)

---

## Lessons Learned

1. **Test filters with real data**: The entity-count filter worked in theory but failed with GLiNER's output distribution
2. **Measure twice, cut once**: Over-pruning deleted 99% of data before we caught it
3. **Simplicity wins**: Removing the filter is better than tuning it
4. **Document decisions**: Future developers need context on why code was removed
5. **Domain-agnostic design**: Financial docs ≠ scientific papers; avoid domain-specific assumptions

---

## Files Reference

**Modified:**

- `indexer.py`
- `graphrag_config.py`

**Created:**

- `project_documentation/contextual_entity_filtering.md`
- `project_documentation/Indexing_troubleshooting_documentation.md` (updated)

**Related:**

- `fix_and_reindex.md` - User-facing recovery guide
- `analyze_pruning.py` - Diagnostic script used during troubleshooting
