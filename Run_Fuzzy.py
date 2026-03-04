r"""
=============================================================================
Sugeno Fuzzy Inference System for SSC Bundle Compliance Assessment
=============================================================================
CONSENSUS CLASSIFICATION:
  - Drug/micro classifications use a SINGLE consensus map built from:
    1. Where regex & gemma AGREE → use that (94.26% drugs, 60.62% micro)
    2. Where they DISAGREE and embedding cosine >= threshold → use gemma
    3. Where they DISAGREE and cosine < threshold → use regex (conservative)
    4. SME overrides applied LAST (always take precedence)

SME OVERRIDES:
  Drug:  Vancomycin Oral Liquid → other (not systemic antibiotic)
         Vancomycin Enema → other (not systemic antibiotic)
  Micro: Viridans streptococci in blood culture → pathogen
         Coagulase-negative staph (all specimens) → pathogen (ICU population)

Usage:
  python fuzzy_compliance.py --sampled_dir ./2000_Patients_Sampled --output_dir ./NeSy_Output
=============================================================================
"""

import json
import os
import time
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLED_DIR   = r"./2000_Patients_Sampled"
OUTPUT_DIR    = r"./NeSy_Output"

REGEX_DRUG_MAP   = os.path.join(OUTPUT_DIR, "regex_drug_map.json")
GEMMA_DRUG_MAP   = os.path.join(OUTPUT_DIR, "gemma_drug_map.json")
REGEX_MICRO_MAP  = os.path.join(OUTPUT_DIR, "regex_micro_map.json")
GEMMA_MICRO_MAP  = os.path.join(OUTPUT_DIR, "gemma_micro_map.json")
EMBEDDING_REPORT = os.path.join(OUTPUT_DIR, "embedding_validation_report.json")

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "fuzzy_compliance_results.csv")

# ==========================================
# MIMIC-IV ITEM IDS
# ==========================================
LACTATE_ITEMID    = 50813
MAP_ITEMIDS       = [220052, 220181, 225312]
CRYSTALLOID_ITEMIDS = [225158, 225159, 220949, 225943, 220862, 225823, 225825, 225827, 225828]
IV_FLUID_CATEGORIES = ['02-Fluids (Crystalloids)', '03-IV Fluid Bolus']
VASO_ITEMIDS      = [221906, 221289, 221749, 222315, 221662, 229617]
VALID_ABX_ROUTES  = {'IV', 'IM', 'IV DRIP', 'IVPB'}

# ==========================================
# FUZZY PARAMETERS (SME consultation)
# ==========================================
SIGMA = {
    'r2_time': 30, 'r3_time': 30,
    'r4_early': 30, 'r4_late': 60,
    'r5_time': 60, 'r5_vol': 10,
    'r6_time': 30, 'r7_map': 5, 'r8_clear': 5,
}

# SME priority weights: R3 > R2 > R5 > R6 > remaining
RULE_WEIGHTS = {1: 0.5, 2: 0.9, 3: 1.0, 4: 0.5, 5: 0.8, 6: 0.7, 7: 0.5, 8: 0.5}

# ==========================================
# SME OVERRIDES
# ==========================================
SME_DRUG_OVERRIDES = {
    'Vancomycin Oral Liquid': 'other',
    'Vancomycin Enema': 'other',
}

# Micro overrides: organism substrings → forced status
# Applied to ALL specimen types
SME_MICRO_ORGANISM_OVERRIDES = {
    'COAGULASE NEGATIVE': 'pathogen',  # ICU population = immunocompromised
}

# Micro overrides: specimen+organism specific
SME_MICRO_SPECIFIC_OVERRIDES = {
    # Viridans streptococci in blood = pathogen (SME)
    # Applied to any key where specimen contains BLOOD and organism contains VIRIDANS
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ==========================================
# MEMBERSHIP FUNCTIONS
# ==========================================
def mu_right(x, c, sigma):
    if x is None or np.isnan(x):
        return None
    return 1.0 if x <= c else float(np.exp(-((x - c)**2) / (2 * sigma**2)))


def mu_left(x, c, sigma):
    if x is None or np.isnan(x):
        return None
    return 1.0 if x >= c else float(np.exp(-((x - c)**2) / (2 * sigma**2)))


def mu_window(x, c1, c2, sigma1, sigma2):
    if x is None or np.isnan(x):
        return None
    if x < c1:
        return float(np.exp(-((x - c1)**2) / (2 * sigma1**2)))
    if x > c2:
        return float(np.exp(-((x - c2)**2) / (2 * sigma2**2)))
    return 1.0


# ==========================================
# DATA HELPERS
# ==========================================
def safe_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def safe_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


# ==========================================
# BUILD CONSENSUS MAPS
# ==========================================
def build_consensus_drug_map(regex_drug, gemma_drug, embedding_report):
    """
    Build single consensus drug classification map:
      1. Agreed → use that
      2. Disagree + embedding cosine ≥ threshold → use gemma (embedding-validated)
      3. Disagree + cosine < threshold → use regex (conservative)
      4. SME overrides applied last
    """
    # Build embedding lookup (keyed by drug name)
    emb_lookup = {}
    for entry in embedding_report.get('confirmed', []) + embedding_report.get('flagged', []):
        emb_lookup[entry['drug']] = entry
        emb_lookup[entry['drug'].strip().upper()] = entry

    all_drugs = set(list(regex_drug.keys()) + list(gemma_drug.keys()))
    consensus = {}
    stats = {'agreed': 0, 'emb_resolved': 0, 'regex_fallback': 0, 'sme_override': 0}

    for drug in all_drugs:
        r_cat = regex_drug.get(drug, 'other')
        g_cat = gemma_drug.get(drug, 'other')

        if r_cat == g_cat:
            consensus[drug] = r_cat
            stats['agreed'] += 1
        else:
            # Check embedding validation
            emb = emb_lookup.get(drug) or emb_lookup.get(drug.strip().upper())
            if emb and emb['similarity'] >= emb['threshold']:
                consensus[drug] = g_cat  # Embedding confirms gemma
                stats['emb_resolved'] += 1
            else:
                consensus[drug] = r_cat  # Conservative regex fallback
                stats['regex_fallback'] += 1

    # Apply SME overrides
    for drug, forced_cat in SME_DRUG_OVERRIDES.items():
        if drug in consensus:
            old = consensus[drug]
            consensus[drug] = forced_cat
            if old != forced_cat:
                stats['sme_override'] += 1
                log(f"  SME drug override: {drug}: {old} → {forced_cat}")

    log(f"  Drug consensus: {stats['agreed']} agreed, {stats['emb_resolved']} emb-resolved, "
        f"{stats['regex_fallback']} regex-fallback, {stats['sme_override']} SME overrides")

    return consensus


def build_consensus_micro_map(regex_micro, gemma_micro):
    """
    Build single consensus micro classification map:
      1. Agreed → use that
      2. is_blood_culture: OR logic (either says True → True)
      3. status: if either says pathogen → pathogen (conservative for compliance)
      4. SME overrides applied last
    """
    all_keys = set(list(regex_micro.keys()) + list(gemma_micro.keys()))
    consensus = {}
    stats = {'agreed': 0, 'resolved': 0, 'sme_override': 0}

    for key in all_keys:
        r_val = regex_micro.get(key, {'is_blood_culture': False, 'status': 'no_growth'})
        g_val = gemma_micro.get(key, {'is_blood_culture': False, 'status': 'no_growth'})

        if r_val == g_val:
            consensus[key] = r_val
            stats['agreed'] += 1
        else:
            # is_blood_culture: OR (conservative — if either detects BC, trust it)
            is_bc = r_val.get('is_blood_culture', False) or g_val.get('is_blood_culture', False)

            # status: prefer pathogen if either says it (conservative for compliance)
            r_s = r_val.get('status', 'no_growth')
            g_s = g_val.get('status', 'no_growth')
            if r_s == g_s:
                status = r_s
            elif 'pathogen' in (r_s, g_s):
                status = 'pathogen'
            else:
                status = r_s  # regex fallback

            consensus[key] = {'is_blood_culture': is_bc, 'status': status}
            stats['resolved'] += 1

    # SME OVERRIDES
    for key in consensus:
        parts = key.split('|')
        if len(parts) < 2:
            continue
        specimen = parts[0].upper()
        organism = parts[1].upper() if parts[1] else ''

        # Viridans streptococci in blood culture = pathogen
        if 'VIRIDANS' in organism and 'BLOOD' in specimen:
            old_status = consensus[key].get('status')
            consensus[key]['status'] = 'pathogen'
            if old_status != 'pathogen':
                stats['sme_override'] += 1

        # Coagulase-negative staph = pathogen in ICU context
        for substr, forced_status in SME_MICRO_ORGANISM_OVERRIDES.items():
            if substr in organism:
                old_status = consensus[key].get('status')
                consensus[key]['status'] = forced_status
                if old_status != forced_status:
                    stats['sme_override'] += 1
                break

    log(f"  Micro consensus: {stats['agreed']} agreed, {stats['resolved']} resolved, "
        f"{stats['sme_override']} SME overrides")

    return consensus


# ==========================================
# FAST LOOKUP BUILDERS
# ==========================================
def build_drug_lookup(drug_map):
    lookup = {}
    for name, cat in drug_map.items():
        lookup[name] = cat
        lookup[name.strip()] = cat
        lookup[name.strip().lower()] = cat
    return lookup


def classify_drug(drug_name, drug_lookup):
    if drug_name is None or (isinstance(drug_name, float) and np.isnan(drug_name)):
        return 'other'
    s = str(drug_name)
    if s in drug_lookup:
        return drug_lookup[s]
    s2 = s.strip()
    if s2 in drug_lookup:
        return drug_lookup[s2]
    s3 = s2.lower()
    if s3 in drug_lookup:
        return drug_lookup[s3]
    return 'other'


def build_micro_lookup(micro_map):
    lookup = {}
    for key, val in micro_map.items():
        lookup[key] = val
        lookup[key.strip()] = val
        lookup[key.upper()] = val
    return lookup


def classify_micro(spec_type, org_name, micro_lookup):
    default = {'is_blood_culture': False, 'status': 'no_growth'}
    if spec_type is None or (isinstance(spec_type, float) and np.isnan(spec_type)):
        return default

    spec = str(spec_type).strip()
    org = str(org_name).strip() if org_name is not None and not (isinstance(org_name, float) and np.isnan(org_name)) else ''

    key = f"{spec}|{org}" if org else f"{spec}|"
    if key in micro_lookup:
        return micro_lookup[key]
    key_upper = key.upper()
    if key_upper in micro_lookup:
        return micro_lookup[key_upper]

    # Specimen-only fallback
    for mk in micro_lookup:
        if mk.startswith(spec + '|') or mk.startswith(spec.upper() + '|'):
            return micro_lookup[mk]

    is_bc = 'blood culture' in spec.lower()
    return {'is_blood_culture': is_bc, 'status': 'no_growth'}


# ==========================================
# EPISODE EVALUATION
# ==========================================
def evaluate_episode(episode_dir, drug_lookup, micro_lookup):
    R = {}
    for r in range(1, 9):
        R[f'rule{r}_score'] = None
        R[f'rule{r}_raw'] = None
        R[f'rule{r}_missing'] = False
        R[f'rule{r}_applicable'] = True if r <= 3 else False
    R['rule5_raw_time'] = None
    R['rule5_raw_vol'] = None

    # Load data
    profile    = safe_json(os.path.join(episode_dir, 'static_profile.json'))
    meds_admin = safe_csv(os.path.join(episode_dir, 'medications_admin.csv'))
    meds_ord   = safe_csv(os.path.join(episode_dir, 'medications_ordered.csv'))
    micro      = safe_csv(os.path.join(episode_dir, 'microbiology.csv'))
    labs       = safe_csv(os.path.join(episode_dir, 'labs.csv'))
    vitals     = safe_csv(os.path.join(episode_dir, 'vitals.csv'))
    inputs     = safe_csv(os.path.join(episode_dir, 'inputs.csv'))

    if not profile:
        R['compliance_score'] = None
        return R

    # ---- t0 ----
    t0 = None
    if profile.get('icu_stays'):
        intimes = [pd.to_datetime(s['intime'], errors='coerce') for s in profile['icu_stays']]
        intimes = [t for t in intimes if pd.notna(t)]
        if intimes:
            t0 = min(intimes)
    if t0 is None:
        t0 = pd.to_datetime(profile.get('admittime'), errors='coerce')
    if t0 is None or pd.isna(t0):
        R['compliance_score'] = None
        return R

    # ---- Weight ----
    weight_kg = None
    if not inputs.empty and 'patientweight' in inputs.columns:
        w = inputs['patientweight'].dropna()
        if not w.empty and w.iloc[0] > 0:
            weight_kg = w.iloc[0]

    # ---- ICU LOS (try icu_stays first, then compute from admit/discharge) ----
    icu_los = None
    if profile.get('icu_stays'):
        los_vals = [s.get('los') for s in profile['icu_stays'] if s.get('los')]
        if los_vals:
            icu_los = sum(los_vals)
    if icu_los is None:
        # Fallback: compute from admittime/dischtime
        admit = pd.to_datetime(profile.get('admittime'), errors='coerce')
        disch = pd.to_datetime(profile.get('dischtime'), errors='coerce')
        if pd.notna(admit) and pd.notna(disch):
            icu_los = (disch - admit).total_seconds() / 86400.0

    R['icu_los'] = icu_los
    R['weight_kg'] = weight_kg

    # ============================================================
    # EXTRACT: First IV/IM antibiotic
    # ============================================================
    t_antibiotic = None
    antibiotic_drug = None

    route_lookup = {}
    if not meds_ord.empty and 'drug' in meds_ord.columns and 'route' in meds_ord.columns:
        for _, row in meds_ord.iterrows():
            d = row.get('drug')
            r_val = row.get('route')
            if pd.notna(d) and pd.notna(r_val):
                route_lookup.setdefault(str(d).strip().lower(), set()).add(str(r_val).strip().upper())

    if not meds_admin.empty and 'medication' in meds_admin.columns:
        meds_admin['charttime'] = pd.to_datetime(meds_admin['charttime'], errors='coerce')
        admin = meds_admin[meds_admin['event_txt'].isin(['Administered', 'Delayed Administered'])]
        admin = admin.dropna(subset=['charttime']).sort_values('charttime')

        for _, row in admin.iterrows():
            drug_name = row.get('medication')
            if pd.isna(drug_name):
                continue
            cat = classify_drug(drug_name, drug_lookup)
            if cat == 'antibiotic':
                drug_lower = str(drug_name).strip().lower()
                routes = route_lookup.get(drug_lower, set())
                is_valid = bool(routes & VALID_ABX_ROUTES) if routes else True
                if is_valid:
                    t_antibiotic = row['charttime']
                    antibiotic_drug = drug_name
                    break

    R['antibiotic_drug'] = antibiotic_drug

    # ============================================================
    # EXTRACT: Blood culture timing
    # ============================================================
    t_culture = None
    has_blood_culture = False

    if not micro.empty and 'spec_type_desc' in micro.columns:
        micro['charttime'] = pd.to_datetime(micro['charttime'], errors='coerce')
        micro_sorted = micro.dropna(subset=['charttime']).sort_values('charttime')

        for _, row in micro_sorted.iterrows():
            spec = row.get('spec_type_desc')
            org = row.get('org_name')
            mc = classify_micro(spec, org, micro_lookup)
            if mc.get('is_blood_culture', False):
                has_blood_culture = True
                if t_culture is None:
                    t_culture = row['charttime']

        if t_culture is None and not micro_sorted.empty:
            t_culture = micro_sorted.iloc[0]['charttime']

    R['has_blood_culture'] = has_blood_culture

    # ============================================================
    # EXTRACT: Lactate values
    # ============================================================
    lac1_val, lac1_time, lac2_val, lac2_time = None, None, None, None

    if not labs.empty and 'itemid' in labs.columns:
        labs['charttime'] = pd.to_datetime(labs['charttime'], errors='coerce')
        lac_rows = labs[labs['itemid'] == LACTATE_ITEMID].dropna(subset=['valuenum', 'charttime'])
        lac_rows = lac_rows.sort_values('charttime')
        if not lac_rows.empty:
            lac1_val = lac_rows.iloc[0]['valuenum']
            lac1_time = lac_rows.iloc[0]['charttime']
            if len(lac_rows) >= 2:
                lac2_val = lac_rows.iloc[1]['valuenum']
                lac2_time = lac_rows.iloc[1]['charttime']

    R['lactate_1_val'] = lac1_val
    R['lactate_2_val'] = lac2_val

    # ============================================================
    # EXTRACT: MAP values
    # ============================================================
    map_df = pd.DataFrame()
    if not vitals.empty and 'itemid' in vitals.columns:
        vitals['charttime'] = pd.to_datetime(vitals['charttime'], errors='coerce')
        map_df = vitals[vitals['itemid'].isin(MAP_ITEMIDS)].dropna(subset=['valuenum', 'charttime'])
        map_df = map_df.sort_values('charttime')

    initial_map = None
    if not map_df.empty:
        early = map_df[(map_df['charttime'] >= t0) & (map_df['charttime'] <= t0 + pd.Timedelta(hours=6))]
        if not early.empty:
            initial_map = early['valuenum'].min()

    R['initial_map'] = initial_map

    # ============================================================
    # EXTRACT: IV Fluid volumes (first 6 hours)
    # ============================================================
    total_fluid_ml = 0.0
    t_fluid_start = None
    t_fluid_end = None

    if not inputs.empty and 'ordercategoryname' in inputs.columns:
        inputs['starttime'] = pd.to_datetime(inputs['starttime'], errors='coerce')
        inputs['endtime'] = pd.to_datetime(inputs['endtime'], errors='coerce')

        cat_match = inputs['ordercategoryname'].isin(IV_FLUID_CATEGORIES)
        id_match = inputs['itemid'].isin(CRYSTALLOID_ITEMIDS)
        fluid_all = inputs[cat_match | id_match].drop_duplicates()

        fluid_win = fluid_all[
            (fluid_all['starttime'] >= t0) &
            (fluid_all['starttime'] <= t0 + pd.Timedelta(hours=6))
        ]

        if not fluid_win.empty and 'amount' in fluid_win.columns:
            ml_mask = fluid_win['amountuom'].astype(str).str.lower().isin(['ml']) | fluid_win['amountuom'].isna()
            total_fluid_ml = fluid_win.loc[ml_mask, 'amount'].sum()
            t_fluid_start = fluid_win['starttime'].min()
            t_fluid_end = fluid_win['endtime'].max()

    R['total_fluid_ml'] = total_fluid_ml

    # ============================================================
    # EXTRACT: Vasopressor timing
    # ============================================================
    t_vasopressor = None
    if not inputs.empty and 'itemid' in inputs.columns:
        vaso_rows = inputs[inputs['itemid'].isin(VASO_ITEMIDS)].sort_values('starttime')
        if not vaso_rows.empty:
            t_vasopressor = pd.to_datetime(vaso_rows.iloc[0]['starttime'], errors='coerce')

    R['has_vasopressor'] = t_vasopressor is not None

    # ============================================================
    # RULE 1: Blood Culture Before Antibiotics (Boolean)
    # ============================================================
    R['rule1_applicable'] = True
    if t_culture is not None and t_antibiotic is not None:
        delta = (t_culture - t_antibiotic).total_seconds() / 60
        R['rule1_raw'] = delta
        R['rule1_score'] = 1.0 if delta < 0 else 0.0
    elif t_culture is None and t_antibiotic is not None:
        R['rule1_score'] = 0.0
        R['rule1_missing'] = True
    elif t_culture is not None and t_antibiotic is None:
        R['rule1_score'] = 0.5
    else:
        R['rule1_score'] = 0.0
        R['rule1_missing'] = True

    # ============================================================
    # RULE 2: Antibiotics Within 1 Hour
    # ============================================================
    R['rule2_applicable'] = True
    if t_antibiotic is not None:
        dt = (t_antibiotic - t0).total_seconds() / 60
        R['rule2_raw'] = dt
        R['rule2_score'] = mu_right(dt, 60, SIGMA['r2_time'])
    else:
        R['rule2_score'] = 0.0
        R['rule2_missing'] = True

    # ============================================================
    # RULE 3: Lactate Within 1 Hour (HIGHEST PRIORITY)
    # ============================================================
    R['rule3_applicable'] = True
    if lac1_time is not None:
        dt = (lac1_time - t0).total_seconds() / 60
        R['rule3_raw'] = dt
        R['rule3_score'] = mu_right(dt, 60, SIGMA['r3_time'])
    else:
        R['rule3_score'] = 0.0
        R['rule3_missing'] = True

    # ============================================================
    # RULE 4: Repeat Lactate if >= 2.0 (2-4 hrs window)
    # ============================================================
    if lac1_val is not None and lac1_val >= 2.0:
        R['rule4_applicable'] = True
        if lac2_time is not None and lac1_time is not None:
            dt = (lac2_time - lac1_time).total_seconds() / 60
            R['rule4_raw'] = dt
            R['rule4_score'] = mu_window(dt, 120, 240, SIGMA['r4_early'], SIGMA['r4_late'])
        else:
            R['rule4_score'] = 0.0
            R['rule4_missing'] = True
    else:
        R['rule4_applicable'] = False
        R['rule4_score'] = 1.0

    # ============================================================
    # RULE 5: Fluid Resuscitation (30 mL/kg, 3 hrs)
    # ============================================================
    trigger_r5 = False
    if initial_map is not None and initial_map < 65:
        trigger_r5 = True
    if lac1_val is not None and lac1_val >= 4.0:
        trigger_r5 = True

    if trigger_r5:
        R['rule5_applicable'] = True
        if t_fluid_start is not None and total_fluid_ml > 0:
            dt = (t_fluid_start - t0).total_seconds() / 60
            R['rule5_raw_time'] = dt
            mu_t = mu_right(dt, 180, SIGMA['r5_time'])
            if weight_kg is not None and weight_kg > 0:
                vpk = total_fluid_ml / weight_kg
                R['rule5_raw_vol'] = vpk
                mu_v = mu_left(vpk, 30, SIGMA['r5_vol'])
                R['rule5_score'] = mu_t * mu_v
            else:
                R['rule5_score'] = mu_t * 0.5
                R['rule5_missing'] = True
        else:
            R['rule5_score'] = 0.0
            R['rule5_missing'] = True if total_fluid_ml == 0 else False
    else:
        if initial_map is None and (lac1_val is None or lac1_val < 4.0):
            R['rule5_applicable'] = False
            R['rule5_score'] = None
        else:
            R['rule5_applicable'] = False
            R['rule5_score'] = 1.0

    # ============================================================
    # RULE 6: Vasopressors if Fluids Fail
    # ============================================================
    post_fluid_map = None
    if t_fluid_end is not None and not map_df.empty:
        pf = map_df[
            (map_df['charttime'] >= t_fluid_end) &
            (map_df['charttime'] <= t_fluid_end + pd.Timedelta(hours=2))
        ]
        if not pf.empty:
            post_fluid_map = pf['valuenum'].min()

    trigger_r6 = False
    if post_fluid_map is not None and post_fluid_map < 65:
        trigger_r6 = True
    elif t_vasopressor is not None and trigger_r5:
        trigger_r6 = True

    if trigger_r6:
        R['rule6_applicable'] = True
        if t_vasopressor is not None:
            ref_time = t_fluid_end if t_fluid_end is not None else t0
            dt = max((t_vasopressor - ref_time).total_seconds() / 60, 0)
            R['rule6_raw'] = dt
            c6 = 60 if t_fluid_end is not None else 180
            R['rule6_score'] = mu_right(dt, c6, SIGMA['r6_time'])
        else:
            R['rule6_score'] = 0.0
            R['rule6_missing'] = True
    else:
        R['rule6_applicable'] = False
        R['rule6_score'] = 1.0

    # ============================================================
    # RULE 7: MAP Recovery >= 65 After Vasopressors
    # ============================================================
    if t_vasopressor is not None and trigger_r6:
        R['rule7_applicable'] = True
        if not map_df.empty:
            pv = map_df[
                (map_df['charttime'] >= t_vasopressor + pd.Timedelta(hours=1)) &
                (map_df['charttime'] <= t_vasopressor + pd.Timedelta(hours=4))
            ]
            if not pv.empty:
                best = pv['valuenum'].max()
                R['rule7_raw'] = best
                R['rule7_score'] = mu_left(best, 65, SIGMA['r7_map'])
            else:
                R['rule7_score'] = None
                R['rule7_missing'] = True
        else:
            R['rule7_score'] = None
            R['rule7_missing'] = True
    else:
        R['rule7_applicable'] = False
        R['rule7_score'] = 1.0

    # ============================================================
    # RULE 8: Lactate Clearance >= 10%
    # ============================================================
    if R['rule4_applicable'] and lac2_val is not None and lac1_val is not None and lac1_val > 0:
        R['rule8_applicable'] = True
        clearance = ((lac1_val - lac2_val) / lac1_val) * 100
        R['rule8_raw'] = clearance
        R['rule8_score'] = mu_left(clearance, 10, SIGMA['r8_clear'])
    else:
        R['rule8_applicable'] = False
        R['rule8_score'] = 1.0

    # ============================================================
    # SUGENO AGGREGATION
    # ============================================================
    num, den, n_eval = 0.0, 0.0, 0
    for r in range(1, 9):
        score = R[f'rule{r}_score']
        appl = R[f'rule{r}_applicable']
        if score is not None and appl:
            w = RULE_WEIGHTS[r]
            num += w * score
            den += w
            n_eval += 1

    R['compliance_score'] = (num / den) if den > 0 else None
    R['evaluable_rules'] = n_eval
    R['t0'] = str(t0)

    return R


# ==========================================
# MAIN
# ==========================================
def main():
    t_start = time.time()
    log("=" * 60)
    log("SUGENO FUZZY COMPLIANCE (CONSENSUS + SME)")
    log("=" * 60)

    # ---- Load raw classification maps ----
    log("Loading classification maps...")
    regex_drug  = safe_json(REGEX_DRUG_MAP)
    gemma_drug  = safe_json(GEMMA_DRUG_MAP)
    regex_micro = safe_json(REGEX_MICRO_MAP)
    gemma_micro = safe_json(GEMMA_MICRO_MAP)
    emb_report  = safe_json(EMBEDDING_REPORT)

    log(f"  Regex drugs: {len(regex_drug)}, Gemma drugs: {len(gemma_drug)}")
    log(f"  Regex micro: {len(regex_micro)}, Gemma micro: {len(gemma_micro)}")

    # ---- Build consensus maps ----
    log("\nBuilding consensus drug map...")
    consensus_drug = build_consensus_drug_map(regex_drug, gemma_drug, emb_report)

    log("\nBuilding consensus micro map...")
    consensus_micro = build_consensus_micro_map(regex_micro, gemma_micro)

    # ---- Build fast lookups ----
    drug_lookup  = build_drug_lookup(consensus_drug)
    micro_lookup = build_micro_lookup(consensus_micro)

    # ---- Enumerate episodes ----
    log(f"\nScanning: {SAMPLED_DIR}")
    episodes = []
    for subject_id in os.listdir(SAMPLED_DIR):
        subject_dir = os.path.join(SAMPLED_DIR, subject_id)
        if not os.path.isdir(subject_dir):
            continue
        for hadm_id in os.listdir(subject_dir):
            episode_dir = os.path.join(subject_dir, hadm_id)
            if not os.path.isdir(episode_dir):
                continue
            if os.path.exists(os.path.join(episode_dir, 'static_profile.json')):
                episodes.append({
                    'subject_id': subject_id,
                    'hadm_id': hadm_id,
                    'path': episode_dir
                })

    log(f"  Found {len(episodes)} episodes")
    log(f"  Unique subjects: {len(set(e['subject_id'] for e in episodes))}")

    # ---- Evaluate with consensus classifications ----
    log("\nEvaluating with CONSENSUS classifications...")
    all_results = []

    for ep in tqdm(episodes, desc="  consensus"):
        try:
            result = evaluate_episode(ep['path'], drug_lookup, micro_lookup)
            result['subject_id'] = ep['subject_id']
            result['hadm_id'] = ep['hadm_id']
            result['classifier'] = 'consensus'
            all_results.append(result)
        except Exception as e:
            all_results.append({
                'subject_id': ep['subject_id'],
                'hadm_id': ep['hadm_id'],
                'classifier': 'consensus',
                'compliance_score': None,
                'error': str(e)
            })

    # ---- Build DataFrame ----
    log("\nBuilding output...")
    df = pd.DataFrame(all_results)

    def worst_rule(row):
        worst_r, worst_s = None, 2.0
        for r in range(1, 9):
            s = row.get(f'rule{r}_score')
            a = row.get(f'rule{r}_applicable', False)
            if a and s is not None and s < worst_s:
                worst_s = s
                worst_r = r
        return worst_r

    df['worst_rule'] = df.apply(worst_rule, axis=1)

    # ---- Save ----
    df.to_csv(OUTPUT_CSV, index=False)
    log(f"\nSaved: {OUTPUT_CSV}")
    log(f"  Total rows: {len(df)}")

    # ---- Summary ----
    scores = df['compliance_score'].dropna()
    if not scores.empty:
        log(f"\n  CONSENSUS ({len(df)} episodes):")
        log(f"    Compliance: mean={scores.mean():.4f}, median={scores.median():.4f}, "
            f"std={scores.std():.4f}, min={scores.min():.4f}, max={scores.max():.4f}")

        log(f"    Per-rule averages:")
        for r in range(1, 9):
            appl = df[df[f'rule{r}_applicable'] == True]
            rs = appl[f'rule{r}_score'].dropna()
            miss = df[f'rule{r}_missing'].sum() if f'rule{r}_missing' in df.columns else 0
            mean_s = f"{rs.mean():.3f}" if not rs.empty else "N/A"
            log(f"      R{r}: mean={mean_s} (n={len(rs)}, missing={int(miss)}, triggered={len(appl)})")

        # Missing data
        log(f"\n    Missing data:")
        for r in range(1, 9):
            m = df[f'rule{r}_missing'].sum() if f'rule{r}_missing' in df.columns else 0
            log(f"      R{r}: {int(m)} ({m/len(df)*100:.1f}%)")

        # Worst rule distribution
        log(f"\n    Worst rule distribution:")
        wr = df['worst_rule'].value_counts().sort_index()
        for r, cnt in wr.items():
            log(f"      R{int(r)}: {cnt} ({cnt/len(df)*100:.1f}%)")

    log(f"\n  Done in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()