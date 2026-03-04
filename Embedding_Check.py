r"""
=============================================================================
Embedding-Based Validation + Disagreement Adjudication
=============================================================================
Two phases:
  Phase 1 (existing): Validate Gemma's non-other classifications using
                       thresholds calibrated from agreed drugs.
  Phase 2 (NEW):      For each disagreed drug, embed the drug
                       and compare cosine similarity to BOTH the regex-claimed
                       anchor AND the gemma-claimed anchor. Higher similarity
                       wins. If one side says "other" (no anchor), check if
                       the other side's similarity passes its threshold.

Usage:
  python embedding_validation.py --model_path /path/to/medgemma.gguf --output_dir ./output
=============================================================================
"""

import json
import numpy as np
import os
import sys
import time
from collections import defaultdict
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = r"./NeSy_Output"
MODEL_PATH = r"medgemma-4b-it-Q4_K_S.gguf"

REGEX_DRUG_MAP = os.path.join(OUTPUT_DIR, "regex_drug_map.json")
GEMMA_DRUG_MAP = os.path.join(OUTPUT_DIR, "gemma_drug_map.json")
EMBED_REPORT   = os.path.join(OUTPUT_DIR, "embedding_validation_report.json")

# Single-word anchors
CATEGORY_ANCHORS = {
    "antibiotic":  "antibiotic",
    "vasopressor": "vasopressor",
    "iv_fluid":    "iv fluid",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def get_embedding(llm, text):
    """Get a single embedding vector by averaging per-token embeddings."""
    raw = llm.embed(text)
    arr = np.array(raw)
    if arr.ndim == 2:
        return arr.mean(axis=0)
    return arr


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def main():
    t0 = time.time()
    log("=" * 60)
    log("EMBEDDING VALIDATION + ADJUDICATION")
    log("=" * 60)

    # ==========================================
    # 1. Load maps
    # ==========================================
    with open(REGEX_DRUG_MAP) as f:
        regex_map = json.load(f)
    with open(GEMMA_DRUG_MAP) as f:
        gemma_map = json.load(f)

    # Separate agreed vs disagreed
    agreed = {}
    disagreed = {}
    for drug in gemma_map:
        if drug in regex_map:
            if gemma_map[drug] == regex_map[drug]:
                agreed[drug] = gemma_map[drug]
            else:
                disagreed[drug] = {
                    'regex': regex_map[drug],
                    'gemma': gemma_map[drug]
                }

    gt_by_cat = defaultdict(list)
    for drug, cat in agreed.items():
        gt_by_cat[cat].append(drug)

    log(f"  Ground truth (both agree): {len(agreed)}")
    for cat, drugs in gt_by_cat.items():
        log(f"    {cat}: {len(drugs)}")
    log(f"  Disagreements: {len(disagreed)}")

    # ==========================================
    # 2. Load model
    # ==========================================
    try:
        from llama_cpp import Llama
    except ImportError:
        log("ERROR: llama-cpp-python not installed")
        sys.exit(1)

    log("\n  Loading model...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_gpu_layers=-1,
        n_batch=512,
        embedding=True,
        verbose=False
    )
    log("  Model loaded.")

    # ==========================================
    # 3. Embed anchors
    # ==========================================
    log("\n  Embedding anchors...")
    anchor_emb = {}
    for cat, phrase in CATEGORY_ANCHORS.items():
        emb = get_embedding(llm, phrase)
        anchor_emb[cat] = emb
        log(f"    {cat}: dim={emb.shape}")

    # ==========================================
    # 4. Calibrate thresholds from agreed drugs
    # ==========================================
    log("\n  Calibrating thresholds from agreed drugs...")
    target_cats = ['antibiotic', 'vasopressor', 'iv_fluid']
    cat_sims = defaultdict(list)

    for cat in target_cats:
        drugs = gt_by_cat.get(cat, [])
        if not drugs:
            log(f"    WARNING: No ground truth for '{cat}'")
            continue

        log(f"    Embedding {len(drugs)} '{cat}' drugs...")
        for drug in tqdm(drugs, desc=f"  {cat}"):
            try:
                drug_emb = get_embedding(llm, drug)
                sim = cosine_similarity(drug_emb, anchor_emb[cat])
                cat_sims[cat].append({'drug': drug, 'sim': sim})
            except Exception as e:
                log(f"      Error '{drug}': {e}")

    thresholds = {}
    for cat in target_cats:
        sims = [x['sim'] for x in cat_sims.get(cat, [])]
        if not sims:
            thresholds[cat] = 0.0
            continue
        mean_s = np.mean(sims)
        std_s = np.std(sims)
        threshold = mean_s - 2 * std_s
        thresholds[cat] = round(float(threshold), 4)

        log(f"\n    {cat}:")
        log(f"      N={len(sims)}, Mean={mean_s:.4f}, Std={std_s:.4f}")
        log(f"      Min={np.min(sims):.4f}, Max={np.max(sims):.4f}")
        log(f"      Threshold (mean-2std): {threshold:.4f}")

        sorted_sims = sorted(cat_sims[cat], key=lambda x: x['sim'])
        log(f"      Bottom 5:")
        for item in sorted_sims[:5]:
            log(f"        {item['sim']:.4f} | {item['drug']}")

    # ==========================================
    # 5. Phase 1: Validate Gemma's non-other classifications
    #    (same as original Embedding_Check.py)
    # ==========================================
    log("\n  PHASE 1: Validating all MedGemma classifications...")
    confirmed = []
    flagged = []
    skipped = []

    for drug in tqdm(list(gemma_map.keys()), desc="Phase 1"):
        gcat = gemma_map[drug]
        if gcat not in target_cats:
            skipped.append({'drug': drug, 'gemma': gcat})
            continue

        try:
            drug_emb = get_embedding(llm, drug)
            sim = cosine_similarity(drug_emb, anchor_emb[gcat])
            entry = {
                'drug': drug,
                'gemma_category': gcat,
                'regex_category': regex_map.get(drug, 'unknown'),
                'similarity': round(sim, 4),
                'threshold': thresholds[gcat]
            }
            if sim >= thresholds[gcat]:
                entry['status'] = 'confirmed'
                confirmed.append(entry)
            else:
                entry['status'] = 'flagged'
                flagged.append(entry)
        except Exception as e:
            flagged.append({
                'drug': drug, 'gemma_category': gcat,
                'status': 'error', 'error': str(e)
            })

    log(f"\n  Phase 1 Results:")
    log(f"    Confirmed: {len(confirmed)}")
    log(f"    Flagged:   {len(flagged)}")
    log(f"    Skipped:   {len(skipped)}")
    log(f"    Rate:      {len(confirmed)/max(len(confirmed)+len(flagged),1)*100:.2f}%")

    # ==========================================
    # 6. Phase 2: Disagreement Adjudication
    #    For each of the 97 disagreed drugs:
    #    - Embed the drug name
    #    - If BOTH sides claim a non-other category:
    #        Compare similarity to regex's anchor vs gemma's anchor
    #        Higher similarity wins
    #    - If one side says "other" (no anchor):
    #        Check if the non-other side's similarity passes threshold
    #        If yes -> non-other side wins
    #        If no  -> "other" side wins
    # ==========================================
    log(f"\n  PHASE 2: Adjudicating {len(disagreed)} disagreements...")

    adjudication_results = []
    adj_summary = {
        'total': len(disagreed),
        'regex_wins': 0,
        'gemma_wins': 0,
        'inconclusive': 0
    }

    for drug, sides in tqdm(disagreed.items(), desc="Phase 2"):
        r_cat = sides['regex']
        g_cat = sides['gemma']

        try:
            drug_emb = get_embedding(llm, drug)
        except Exception as e:
            adjudication_results.append({
                'drug': drug, 'regex': r_cat, 'gemma': g_cat,
                'winner': 'error', 'reason': str(e)
            })
            adj_summary['inconclusive'] += 1
            continue

        r_has_anchor = r_cat in target_cats
        g_has_anchor = g_cat in target_cats

        # Case 1: Both have anchors (e.g., regex=vasopressor, gemma=antibiotic)
        if r_has_anchor and g_has_anchor:
            r_sim = cosine_similarity(drug_emb, anchor_emb[r_cat])
            g_sim = cosine_similarity(drug_emb, anchor_emb[g_cat])

            if r_sim > g_sim:
                winner = 'regex'
                adj_summary['regex_wins'] += 1
            elif g_sim > r_sim:
                winner = 'gemma'
                adj_summary['gemma_wins'] += 1
            else:
                winner = 'tie'
                adj_summary['inconclusive'] += 1

            adjudication_results.append({
                'drug': drug,
                'regex': r_cat, 'regex_sim': round(r_sim, 4),
                'gemma': g_cat, 'gemma_sim': round(g_sim, 4),
                'winner': winner,
                'reason': f"both_anchored: regex_sim={r_sim:.4f} vs gemma_sim={g_sim:.4f}"
            })

        # Case 2: Regex says non-other, Gemma says other
        elif r_has_anchor and not g_has_anchor:
            r_sim = cosine_similarity(drug_emb, anchor_emb[r_cat])
            if r_sim >= thresholds[r_cat]:
                winner = 'regex'
                adj_summary['regex_wins'] += 1
                reason = f"regex_anchored: sim={r_sim:.4f} >= threshold={thresholds[r_cat]}"
            else:
                winner = 'gemma'
                adj_summary['gemma_wins'] += 1
                reason = f"regex_below_threshold: sim={r_sim:.4f} < threshold={thresholds[r_cat]}"

            adjudication_results.append({
                'drug': drug,
                'regex': r_cat, 'regex_sim': round(r_sim, 4),
                'gemma': g_cat, 'gemma_sim': None,
                'winner': winner, 'reason': reason
            })

        # Case 3: Gemma says non-other, Regex says other
        elif g_has_anchor and not r_has_anchor:
            g_sim = cosine_similarity(drug_emb, anchor_emb[g_cat])
            if g_sim >= thresholds[g_cat]:
                winner = 'gemma'
                adj_summary['gemma_wins'] += 1
                reason = f"gemma_anchored: sim={g_sim:.4f} >= threshold={thresholds[g_cat]}"
            else:
                winner = 'regex'
                adj_summary['regex_wins'] += 1
                reason = f"gemma_below_threshold: sim={g_sim:.4f} < threshold={thresholds[g_cat]}"

            adjudication_results.append({
                'drug': drug,
                'regex': r_cat, 'regex_sim': None,
                'gemma': g_cat, 'gemma_sim': round(g_sim, 4),
                'winner': winner, 'reason': reason
            })

        # Case 4: Both say other (shouldn't happen in disagreements, but handle)
        else:
            adjudication_results.append({
                'drug': drug,
                'regex': r_cat, 'gemma': g_cat,
                'winner': 'inconclusive',
                'reason': 'both_other_or_unknown'
            })
            adj_summary['inconclusive'] += 1

    # ==========================================
    # 7. Build final report
    # ==========================================
    report = {
        'phase1_summary': {
            'total_drugs': len(gemma_map),
            'ground_truth': len(agreed),
            'validated_non_other': len(confirmed) + len(flagged),
            'confirmed': len(confirmed),
            'flagged': len(flagged),
            'skipped_other': len(skipped),
            'confirmation_rate': round(
                len(confirmed) / max(len(confirmed) + len(flagged), 1) * 100, 2)
        },
        'thresholds': thresholds,
        'calibration': {
            cat: {
                'count': len(cat_sims.get(cat, [])),
                'mean': round(float(np.mean([x['sim'] for x in cat_sims[cat]])), 4),
                'std': round(float(np.std([x['sim'] for x in cat_sims[cat]])), 4),
                'min': round(float(np.min([x['sim'] for x in cat_sims[cat]])), 4),
                'max': round(float(np.max([x['sim'] for x in cat_sims[cat]])), 4),
                'threshold': thresholds.get(cat, 0)
            }
            for cat in target_cats if cat_sims.get(cat)
        },
        'phase2_adjudication_summary': adj_summary,
        'phase2_adjudication_details': sorted(
            adjudication_results, key=lambda x: x['drug']
        ),
        'phase1_confirmed': sorted(confirmed, key=lambda x: x['similarity']),
        'phase1_flagged': sorted(flagged, key=lambda x: x.get('similarity', 0)),
    }

    with open(EMBED_REPORT, 'w') as f:
        json.dump(report, f, indent=2)

    # ==========================================
    # 8. Print results
    # ==========================================
    log(f"\n{'='*60}")
    log(f"PHASE 1 RESULTS (Gemma validation)")
    log(f"{'='*60}")
    log(f"  Confirmed:         {len(confirmed)}")
    log(f"  Flagged:           {len(flagged)}")
    log(f"  Confirmation rate: {report['phase1_summary']['confirmation_rate']}%")

    if flagged:
        log(f"\n  FLAGGED DRUGS:")
        for f_entry in sorted(flagged, key=lambda x: x.get('similarity', 0)):
            log(f"    sim={f_entry.get('similarity','?'):>7} thresh={f_entry.get('threshold','?')} | "
                f"gemma={f_entry['gemma_category']:12s} regex={f_entry.get('regex_category','?'):12s} | "
                f"{f_entry['drug']}")

    log(f"\n{'='*60}")
    log(f"PHASE 2 RESULTS (Disagreement Adjudication)")
    log(f"{'='*60}")
    log(f"  Total disagreements: {adj_summary['total']}")
    log(f"  Regex wins:          {adj_summary['regex_wins']}")
    log(f"  Gemma wins:          {adj_summary['gemma_wins']}")
    log(f"  Inconclusive:        {adj_summary['inconclusive']}")

    log(f"\n  ADJUDICATION DETAILS:")
    for entry in sorted(adjudication_results, key=lambda x: x.get('winner', '')):
        r_sim_str = f"{entry.get('regex_sim', 'N/A'):>7}" if entry.get('regex_sim') is not None else "  N/A  "
        g_sim_str = f"{entry.get('gemma_sim', 'N/A'):>7}" if entry.get('gemma_sim') is not None else "  N/A  "
        log(f"    {entry['winner']:>6} | r_sim={r_sim_str} g_sim={g_sim_str} | "
            f"regex={entry['regex']:12s} gemma={entry['gemma']:12s} | {entry['drug']}")

    log(f"\n  Saved: {EMBED_REPORT}")
    log(f"  Done in {(time.time()-t0)/60:.1f} min")

    del llm
    import gc; gc.collect()


if __name__ == "__main__":
    main()