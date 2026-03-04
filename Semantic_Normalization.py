r"""
=============================================================================
Semantic Sepsis Pipeline
=============================================================================
Fixes from validation analysis:

REGEX FIXES:
  - Exclude topical/ophthalmic/otic/nasal formulations from systemic antibiotics
  - Add missing antibiotics: amoxicillin, cephalexin, ofloxacin, moxifloxacin,
    ertapenem, fidaxomicin, omadacycline, cefiderocol, dicloxacillin, penicillins,
    nitrofurantoin, remdesivir, sulfadiazine
  - Handle MIMIC tall-man lettering: MetroNIDAZOLE, CefePIME, ValACYclovir,
    DAPTOmycin, NOREPINEPHrine, PHENYLEPHrine, etc.
  - Exclude Lidocaine/Epinephrine combos (local anesthetic, not vasopressor)
  - Exclude nasal phenylephrine spray (not systemic vasopressor)
  - Albumin → iv_fluid (colloid resuscitant)
  - Exclude flush syringes, epidural bags, intrapleural, inhalation from iv_fluid
  - Exclude TPN (Amino Acids + Dextrose) from iv_fluid
  - Handle truncated drug names: NORepinephri, ValGANCIclo, ValGANCIc, etc.
  - Dextrose 5% plain → iv_fluid (unless in TPN or special formulation)

GEMMA PROMPT FIXES:
  - Explicit list of NON-blood-culture specimens
  - Better pathogen vs contaminant guidance
  - Force status to lowercase enum
  - Handle "UNKNOWN" organism as no_growth

Pipeline: Extract → Regex → MedGemma → Validate → Build JSONs

Usage:
  python semantic_pipeline.py --data_dir ./2000_Patients_Sampled --output_dir ./NeSy_Output
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import json
import re
import sys
import gc
import time
from collections import defaultdict
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = r"./2000_Patients_Sampled"
OUTPUT_DIR = r"./NeSy_Output"
MODEL_PATH = r"medgemma-4b-it-Q4_K_S.gguf"

UNIQUE_STRINGS_FILE   = os.path.join(OUTPUT_DIR, "unique_strings.json")
REGEX_DRUG_MAP_FILE   = os.path.join(OUTPUT_DIR, "regex_drug_map.json")
REGEX_MICRO_MAP_FILE  = os.path.join(OUTPUT_DIR, "regex_micro_map.json")
GEMMA_DRUG_MAP_FILE   = os.path.join(OUTPUT_DIR, "gemma_drug_map.json")
GEMMA_MICRO_MAP_FILE  = os.path.join(OUTPUT_DIR, "gemma_micro_map.json")
VALIDATION_FILE       = os.path.join(OUTPUT_DIR, "validation_report.json")

# MIMIC-IV Known ItemIDs
LACTATE_ITEMIDS = {50813, 52442}
VITAL_ITEMIDS = {
    220052: "MAP", 220181: "MAP",
    220179: "SBP", 220180: "DBP",
    220045: "HR", 220277: "SpO2",
    220210: "RR",
    223761: "Temp_F", 223762: "Temp_C",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ==========================================
# CORRECTED REGEX / DICTIONARY BASELINE
# ==========================================

# --- EXCLUSION PATTERNS (applied BEFORE classification) ---
# Topical, ophthalmic, otic, nasal, rectal formulations = NOT systemic
TOPICAL_RE = re.compile(
    r'\b(?:topical|cream|ointment|oint|gel|lotion|ophth(?:almic)?|'
    r'otic|nasal\s*spray|rectal|vaginal|opht\.?\s*(?:soln|susp|oint)|'
    r'ophth\s*oint|opth)\b',
    re.IGNORECASE
)

# Local anesthetic combos with epinephrine (NOT vasopressors)
LOCAL_ANESTHETIC_EPINEPHRINE_RE = re.compile(
    r'\b(?:lidocaine|bupivacaine|ropivacaine|mepivacaine)\b.*\b(?:epinephrine|epi)\b',
    re.IGNORECASE
)

# Non-resuscitation fluid patterns
NON_RESUSCITATION_RE = re.compile(
    r'\b(?:flush|syringe|epidural|intrapleural|inhalation|'
    r'amino\s*acids?.*dextrose|TPN|parenteral\s*nutrition|'
    r'potassium\s*chl.*D5W|glass\s*bottle)\b',
    re.IGNORECASE
)

# --- ANTIBIOTICS ---
ANTIBIOTIC_PATTERNS = [
    # Penicillins
    r'\bamoxicillin\b', r'\baugmentin\b', r'\bamoxicillin[\-\s]*clavulan',
    r'\bampicillin\b', r'\bsulbactam\b', r'\bunasyn\b',
    r'\bpenicillin\b', r'\bdicloxacillin\b', r'\boxacillin\b', r'\bnafcillin\b',
    r'\bpiperacillin\b', r'\btazobactam\b', r'\bzosyn\b', r'\bpip[\-/]?tazo?\b',

    # Cephalosporins
    r'\bcefazolin\b', r'\bancef\b',
    r'\bcephalexin\b', r'\bkeflex\b',
    r'\bcefuroxime\b', r'\bzinacef\b',
    r'\bceftriaxone\b', r'\brocephin\b',
    r'\bceftazidime\b', r'\bfortaz\b',
    r'\bcefepime\b', r'\bcefepim\b', r'\bmaxipime\b',
    r'\bcefpodoxime\b', r'\bvantin\b',
    r'\bceftaroline\b', r'\bteflaro\b',
    r'\bcefiderocol\b', r'\bcefidero\b',

    # Carbapenems
    r'\bmeropenem\b', r'\bmerrem\b',
    r'\bertapenem\b', r'\binvanz\b',
    r'\bimipenem\b', r'\bcilastatin\b', r'\bprimaxin\b',
    r'\bdoripenem\b',

    # Fluoroquinolones
    r'\blevofloxacin\b', r'\blevaquin\b',
    r'\bciprofloxacin\b', r'\bcipro\b',
    r'\bmoxifloxacin\b', r'\bavelox\b', r'\bvigamox\b',
    r'\bofloxacin\b',

    # Glycopeptides / Lipopeptides
    r'\bvancomycin\b', r'\bvancocin\b', r'\bvanco\b',
    r'\bdaptomycin\b', r'\bcubicin\b',

    # Aminoglycosides
    r'\bgentamicin\b', r'\bgaramycin\b',
    r'\btobramycin\b',
    r'\bamikacin\b',

    # Macrolides / Tetracyclines
    r'\bazithromycin\b', r'\bzithromax\b', r'\bz[\-]?pak\b',
    r'\berythromycin\b',
    r'\bdoxycycline\b',
    r'\bomadacycline\b',
    r'\btigecycline\b', r'\btygacil\b',

    # Other antibacterials
    r'\bmetronidazole\b', r'\bflagyl\b',
    r'\blinezolid\b', r'\bzyvox\b',
    r'\btrimethoprim\b', r'\bsulfamethoxazole\b', r'\bbactrim\b',
    r'\bclindamycin\b', r'\bcleocin\b',
    r'\baztreonam\b',
    r'\bcolistin\b', r'\bpolymyxin\b',
    r'\brifampin\b', r'\brifampicin\b',
    r'\bnitrofurantoin\b', r'\bmacrobid\b',
    r'\bfidaxomicin\b', r'\bdificid\b',
    r'\bsulfadiazine\b',

    # Antifungals (systemic)
    r'\bfluconazole\b', r'\bdiflucan\b',
    r'\bmicafungin\b', r'\bmycamine\b',
    r'\bcaspofungin\b', r'\bcancidas\b',
    r'\bamphotericin\b', r'\bambisome\b',
    r'\bvoriconazole\b', r'\bvfend\b',

    # Antivirals (systemic)
    r'\bacyclovir\b', r'\bvalacyclovir\b', r'\bvaltrex\b',
    r'\bganciclovir\b', r'\bvalganciclovir\b',
    r'\bremdesivir\b',

    # MIMIC Tall-Man Lettering (case-insensitive handles this, but explicit)
    r'\bMetroNIDAZOLE\b', r'\bCefePIME\b', r'\bValACYclovir\b',
    r'\bDAPTOmycin\b', r'\bRifAMPin\b', r'\bCeFAZolin\b',
    r'\bValGANCIclo\w*\b',  # Catches ValGANCIclo, ValGANCIclovir, ValGANCIc

    # Truncated names from MIMIC
    r'\bValGANCIcl\b', r'\bValGANCIc\b',
]
ANTIBIOTIC_RE = re.compile('|'.join(ANTIBIOTIC_PATTERNS), re.IGNORECASE)

# --- VASOPRESSORS ---
VASOPRESSOR_PATTERNS = [
    r'\bnorepinephrine\b', r'\blevophed\b',
    r'\bvasopressin\b',
    r'\bphenylephrine\b', r'\bneosynephrine\b', r'\bneo[\-]?synephrine\b',
    r'\bepinephrine\b', r'\bracepinephrine\b',
    r'\bdopamine\b',
    r'\bdobutamine\b',
    r'\bmilrinone\b', r'\bprimacor\b',
    r'\bangiotensin\s*II\b', r'\bangiotensin\b',

    # MIMIC Tall-Man + Truncated
    r'\bNOREPINEPHri\w*\b',  # NORepinephri, NOREPINEPHrine
    r'\bPHENYLEPHri\w*\b',   # PHENYLEPHrine, PHENYLEPHrin
]
VASOPRESSOR_RE = re.compile('|'.join(VASOPRESSOR_PATTERNS), re.IGNORECASE)

# --- IV FLUIDS ---
IV_FLUID_PATTERNS = [
    r'\blactated\s*ringer', r'\b(?:LR|RL)\b',
    r'\bnormal\s*saline\b',
    r'\b0\.?9\s*%?\s*(?:sodium\s*chloride|nacl)\b',
    r'\bplasmalyte\b', r'\bplasma[\-\s]?lyte\b',
    r'\bD5\s*W\b', r'\bD5\s*NS\b', r'\bD5\s*(?:1/?2|1/?4)\s*NS\b', r'\bD5\s*LR\b',
    r'\bdextrose\s+5\s*%\b',
    r'\b0\.?45\s*%?\s*(?:sodium\s*chloride|nacl)\b',
    r'\bhalf[\-\s]*normal\s*saline\b',
    r'\bcrystalloid\b',
    r'\bIso[\-\s]*Osmotic\s*(?:Dextrose|Sodium\s*Chloride)\b',
    r'\bIsotonic\s*Sodium\s*Chloride\b',
    r'\b(?:Dextrose\s*5\s*%)\b',  # Plain D5W

    # Sodium Chloride with concentrations (resuscitation range)
    r'\bSodium\s*Chloride\s*(?:0\.9|0\.45)\b',
    r'\b(?:0\.9|0\.45)\s*%?\s*Sodium\s*Chloride\b',

    # Albumin (colloid resuscitant)
    r'\bAlbumin\b',

    # Prismasate / CRRT solutions
    r'\bPrismasate\b',

    # Generic "Sodium Chloride" without qualifier — likely NS
    r'\bSodium\s+Chloride\s*$',
    r'^Sodium\s+Chloride$',
    r'^sodium\s+chloride$',
]
IV_FLUID_RE = re.compile('|'.join(IV_FLUID_PATTERNS), re.IGNORECASE)

# Additional exclusion: hypertonic saline (3%, 23.4%) is NOT resuscitation
HYPERTONIC_RE = re.compile(r'\b(?:3|23\.4|7\.5)\s*%\s*(?:Sodium\s*Chloride|NaCl)', re.IGNORECASE)

# Flush/syringe patterns
FLUSH_RE = re.compile(r'\bflush\b|\bsyringe\b', re.IGNORECASE)

# --- MICROBIOLOGY ---
BLOOD_CULTURE_RE = re.compile(
    r'\bblood\s*culture\b|'
    r'\bfluid\s*received\s*in\s*blood\s*culture\s*bottle',
    re.IGNORECASE
)

# Explicit NON-blood-culture specimens
NON_BLOOD_CULTURE_RE = re.compile(
    r'\b(?:sputum|urine|stool|swab|abscess|tissue|wound|'
    r'catheter\s*tip|bronchoalveolar|BAL|pleural\s*fluid|'
    r'peritoneal|csf|spinal\s*fluid|foreign\s*body|'
    r'mrsa\s*screen|rectal|nasal|throat)\b',
    re.IGNORECASE
)

# Contaminants
CONTAMINANT_PATTERNS = [
    r'\bcoag[\-\s]*neg', r'\bcoagulase[\-\s]*negative\b',
    r'\bstaphylococcus\s+epidermidis\b',
    r'\bcorynebacterium\b',
    r'\bbacillus\s+(?:species|sp)\b(?!.*anthracis)',
    r'\bpropionibacterium\b', r'\bcutibacterium\b',
    r'\bmicrococcus\b',
    r'\blactobacillus\b',
]
CONTAMINANT_RE = re.compile('|'.join(CONTAMINANT_PATTERNS), re.IGNORECASE)

# Viridans strep — contaminant in single blood culture
VIRIDANS_RE = re.compile(r'\bviridans\b', re.IGNORECASE)

# Mixed/normal flora
MIXED_FLORA_RE = re.compile(r'\bmixed\s*(?:bacterial\s*)?flora\b|\bnormal\s*flora\b', re.IGNORECASE)

NO_GROWTH_RE = re.compile(
    r'\bno\s*growth\b|\bnegative\b|\bno\s*organism\b|\bno\s*isolate',
    re.IGNORECASE
)


# ==========================================
# CLASSIFICATION FUNCTIONS
# ==========================================

def regex_classify_drug(drug_name):
    """Classify a drug using corrected regex with exclusions."""
    name = str(drug_name).strip()
    if not name or name.lower() in ('nan', 'none', ''):
        return 'unknown'

    is_topical = bool(TOPICAL_RE.search(name))
    is_local_epi = bool(LOCAL_ANESTHETIC_EPINEPHRINE_RE.search(name))
    is_flush = bool(FLUSH_RE.search(name))
    is_non_resus = bool(NON_RESUSCITATION_RE.search(name))
    is_hypertonic = bool(HYPERTONIC_RE.search(name))
    is_nasal_phenyl = bool(re.search(r'phenylephrine.*nasal|nasal.*phenylephrine', name, re.I))

    # 1. ANTIBIOTIC — but NOT topical/ophthalmic formulations
    if ANTIBIOTIC_RE.search(name) and not is_topical:
        # Special case: "Desensitization" protocols are still antibiotics
        return 'antibiotic'

    # 2. VASOPRESSOR — but NOT local anesthetic combos or nasal spray
    if VASOPRESSOR_RE.search(name) and not is_local_epi and not is_nasal_phenyl:
        # Check if it's a vasopressor mixed into an IV bag (e.g., PHENYLEPHrin 60mg/250mL NS)
        # These ARE vasopressors (IV drip form)
        return 'vasopressor'

    # 3. IV FLUID — but NOT flushes, syringes, epidural, TPN, inhalation, hypertonic
    if IV_FLUID_RE.search(name) and not is_flush and not is_non_resus and not is_hypertonic:
        return 'iv_fluid'

    return 'other'


def regex_classify_micro(spec_type, org_name):
    """Classify microbiology using corrected regex."""
    spec = str(spec_type).strip() if pd.notna(spec_type) else ''
    org = str(org_name).strip() if pd.notna(org_name) else ''

    # Blood culture detection
    if BLOOD_CULTURE_RE.search(spec):
        is_bc = True
    elif NON_BLOOD_CULTURE_RE.search(spec):
        is_bc = False
    else:
        # Unknown specimen — default to not blood culture
        is_bc = False

    # Organism status
    if not org or org.lower() in ('nan', 'none', '', 'unknown'):
        status = 'no_growth'
    elif NO_GROWTH_RE.search(org):
        status = 'no_growth'
    elif CONTAMINANT_RE.search(org):
        status = 'contaminant'
    elif VIRIDANS_RE.search(org):
        # Viridans strep in single culture = likely contaminant
        status = 'contaminant'
    elif MIXED_FLORA_RE.search(org):
        status = 'contaminant'
    else:
        status = 'pathogen'

    return {'is_blood_culture': is_bc, 'status': status}


def regex_classify_input(ordercategory):
    """Classify inputevents by ordercategoryname."""
    cat = str(ordercategory).strip().lower() if pd.notna(ordercategory) else ''
    if 'antibiotic' in cat:
        return 'antibiotic'
    if 'fluid' in cat or ('bolus' in cat and 'med' not in cat):
        return 'iv_fluid'
    if 'drip' in cat:
        return 'drip'
    if 'blood product' in cat:
        return 'blood_product'
    return 'other'


# ==========================================
# STEP 1: EXTRACT UNIQUE STRINGS
# ==========================================

def extract_unique_strings():
    log("STEP 1: Extracting unique strings...")

    if os.path.exists(UNIQUE_STRINGS_FILE):
        log(f"  Cached: {UNIQUE_STRINGS_FILE}")
        with open(UNIQUE_STRINGS_FILE, 'r') as f:
            data = json.load(f)
        log(f"  {len(data['drugs'])} drugs, {len(data['micro'])} micro")
        return data['drugs'], data['micro']

    unique_drugs = set()
    unique_micro = set()

    subjects = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    for subj_id in tqdm(subjects, desc="Scanning"):
        subj_path = os.path.join(DATA_DIR, subj_id)
        for hadm_id in os.listdir(subj_path):
            ep_path = os.path.join(subj_path, hadm_id)
            if not os.path.isdir(ep_path):
                continue

            for file, col in [('medications_admin.csv', 'medication'),
                               ('medications_ordered.csv', 'drug')]:
                fpath = os.path.join(ep_path, file)
                if os.path.exists(fpath):
                    try:
                        df = pd.read_csv(fpath)
                        if col in df.columns:
                            unique_drugs.update(df[col].dropna().astype(str).unique())
                    except Exception:
                        pass

            micro_path = os.path.join(ep_path, 'microbiology.csv')
            if os.path.exists(micro_path):
                try:
                    df = pd.read_csv(micro_path)
                    for _, row in df.iterrows():
                        spec = str(row.get('spec_type_desc', 'UNKNOWN'))
                        org = str(row.get('org_name', 'UNKNOWN'))
                        if spec == 'nan': spec = 'UNKNOWN'
                        if org == 'nan': org = 'UNKNOWN'
                        unique_micro.add(f"{spec}|{org}")
                except Exception:
                    pass

    drugs_list = sorted(unique_drugs)
    micro_list = sorted(unique_micro)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(UNIQUE_STRINGS_FILE, 'w') as f:
        json.dump({'drugs': drugs_list, 'micro': micro_list}, f, indent=2)

    log(f"  {len(drugs_list)} drugs, {len(micro_list)} micro")
    return drugs_list, micro_list


# ==========================================
# STEP 2: REGEX BASELINE
# ==========================================

def run_regex_baseline(drugs, micro_events):
    log("\nSTEP 2: Regex baseline...")

    # Drugs
    regex_drug_map = {}
    for drug in tqdm(drugs, desc="Regex drugs"):
        regex_drug_map[drug] = regex_classify_drug(drug)
    with open(REGEX_DRUG_MAP_FILE, 'w') as f:
        json.dump(regex_drug_map, f, indent=2)

    # Micro
    regex_micro_map = {}
    for combo in tqdm(micro_events, desc="Regex micro"):
        parts = combo.split('|', 1)
        spec = parts[0] if len(parts) > 0 else 'UNKNOWN'
        org = parts[1] if len(parts) > 1 else 'UNKNOWN'
        regex_micro_map[combo] = regex_classify_micro(spec, org)
    with open(REGEX_MICRO_MAP_FILE, 'w') as f:
        json.dump(regex_micro_map, f, indent=2)

    # Summary
    dcounts = defaultdict(int)
    for v in regex_drug_map.values():
        dcounts[v] += 1
    log(f"  Drugs: {dict(dcounts)}")

    bc_count = sum(1 for v in regex_micro_map.values() if v.get('is_blood_culture'))
    log(f"  Blood cultures: {bc_count}/{len(regex_micro_map)}")

    return regex_drug_map, regex_micro_map


# ==========================================
# STEP 3: MEDGEMMA CLASSIFICATION
# ==========================================

def run_medgemma(drugs, micro_events):
    log("\nSTEP 3: MedGemma classification...")

    try:
        from llama_cpp import Llama
    except ImportError:
        log("  llama-cpp-python not installed. Skipping.")
        return None, None

    if not os.path.exists(MODEL_PATH):
        log(f"  Model not found: {MODEL_PATH}")
        return None, None

    log(f"  Loading model...")
    llm = Llama(model_path=MODEL_PATH, n_ctx=4096, n_gpu_layers=-1,
                n_batch=512, verbose=False)
    log("  Model loaded.")

    # --- DRUGS ---
    gemma_drug_map = {}
    if os.path.exists(GEMMA_DRUG_MAP_FILE):
        with open(GEMMA_DRUG_MAP_FILE, 'r') as f:
            gemma_drug_map = json.load(f)
        log(f"  Resuming drugs: {len(gemma_drug_map)} done")

    remaining = [d for d in drugs if d not in gemma_drug_map]
    log(f"  Classifying {len(remaining)} drugs...")

    for i, drug in enumerate(tqdm(remaining, desc="Gemma drugs")):
        prompt = f"""<start_of_turn>user
You are an expert clinical pharmacist classifying medications for a sepsis treatment protocol.

RULES:
- antibiotic: SYSTEMIC antibiotics, antifungals, antivirals ONLY. Excludes topical creams, ophthalmic drops, otic solutions, nasal sprays.
- vasopressor: INTRAVENOUS vasopressors/inotropes only. Excludes local anesthetic combos (Lidocaine+Epinephrine), nasal sprays (Phenylephrine nasal), ophthalmic drops.
- iv_fluid: IV crystalloids/colloids for VOLUME RESUSCITATION. Includes Normal Saline, Lactated Ringers, PlasmaLyte, Albumin, D5W. Excludes flushes, syringes, TPN, inhalation solutions, hypertonic saline (3%, 23.4%).
- other: Everything else (PPIs, sedatives, insulin, vitamins, diuretics, local anesthetics, chemotherapy, vaccines, electrolyte replacements).

IMPORTANT:
- Topical Metronidazole/Gentamicin/Erythromycin = other (not systemic)
- Lidocaine + Epinephrine = other (local anesthetic)
- Phenylephrine nasal spray = other (not systemic vasopressor)
- Torsemide/Furosemide = other (diuretics, not IV fluids)
- Sterile Water for injection = other (not resuscitation fluid)
- Daunorubicin/Idarubicin = other (chemotherapy, not antibiotics)
- Vaccines = other

Medication: "{drug}"

Respond with ONLY one word: antibiotic, vasopressor, iv_fluid, or other<end_of_turn>
<start_of_turn>model
"""
        try:
            output = llm(prompt, max_tokens=10, stop=["<end_of_turn>", "\n"],
                         echo=False, temperature=0)
            cat = output['choices'][0]['text'].strip().lower().rstrip('.')
            if cat not in ('antibiotic', 'vasopressor', 'iv_fluid', 'other'):
                cat = 'other'
        except Exception as e:
            log(f"    Error: {drug}: {e}")
            cat = 'error'

        gemma_drug_map[drug] = cat
        if (i + 1) % 50 == 0:
            with open(GEMMA_DRUG_MAP_FILE, 'w') as f:
                json.dump(gemma_drug_map, f, indent=2)

    with open(GEMMA_DRUG_MAP_FILE, 'w') as f:
        json.dump(gemma_drug_map, f, indent=2)

    # --- MICRO ---
    gemma_micro_map = {}
    if os.path.exists(GEMMA_MICRO_MAP_FILE):
        with open(GEMMA_MICRO_MAP_FILE, 'r') as f:
            gemma_micro_map = json.load(f)
        log(f"  Resuming micro: {len(gemma_micro_map)} done")

    remaining_m = [m for m in micro_events if m not in gemma_micro_map]
    log(f"  Classifying {len(remaining_m)} micro events...")

    for i, combo in enumerate(tqdm(remaining_m, desc="Gemma micro")):
        parts = combo.split('|', 1)
        spec = parts[0] if len(parts) > 0 else 'UNKNOWN'
        org = parts[1] if len(parts) > 1 else 'UNKNOWN'

        prompt = f"""<start_of_turn>user
You are a clinical microbiologist evaluating specimens for a sepsis blood culture protocol.

QUESTION 1 - Is this a BLOOD CULTURE?
- ONLY these are blood cultures: "BLOOD CULTURE", "FLUID RECEIVED IN BLOOD CULTURE BOTTLES"
- These are NOT blood cultures: SPUTUM, URINE, STOOL, SWAB, ABSCESS, TISSUE, WOUND, CATHETER TIP, BRONCHOALVEOLAR LAVAGE, PLEURAL FLUID, PERITONEAL FLUID, CSF, FOREIGN BODY, MRSA SCREEN, FLUID OTHER
- If unsure, answer false.

QUESTION 2 - Organism status:
- pathogen: S. aureus, E. coli, Klebsiella, Pseudomonas, Candida (any species), Enterococcus, Strep pneumoniae, Serratia, Proteus, Acinetobacter, Enterobacter, Citrobacter, Aeromonas, Nocardia, Stenotrophomonas, Mycobacterium, Gram negative rods, Gram positive cocci
- contaminant: Coagulase-negative Staphylococcus, S. epidermidis, Corynebacterium, Bacillus (not anthracis), Cutibacterium/Propionibacterium, Micrococcus, Viridans streptococci, Lactobacillus, Mixed bacterial flora, Normal flora
- no_growth: If organism is empty, "UNKNOWN", "No Growth", "Negative", or no organism isolated

Specimen: "{spec}"
Organism: "{org}"

Respond with ONLY valid JSON:
{{"is_blood_culture": true_or_false, "status": "pathogen_or_contaminant_or_no_growth"}}<end_of_turn>
<start_of_turn>model
"""
        try:
            output = llm(prompt, max_tokens=64, stop=["<end_of_turn>"],
                         echo=False, temperature=0)
            text = output['choices'][0]['text'].strip()
            json_str = text[text.find('{'):text.rfind('}') + 1]
            result = json.loads(json_str)

            # Normalize status
            if 'status' in result:
                result['status'] = result['status'].strip().lower()
                if result['status'] not in ('pathogen', 'contaminant', 'no_growth'):
                    if 'unknown' in result['status'].lower() or result['status'] == '':
                        result['status'] = 'no_growth'
                    else:
                        result['status'] = 'no_growth'
            else:
                result['status'] = 'no_growth'

            if 'is_blood_culture' not in result:
                result['is_blood_culture'] = False

        except Exception:
            result = regex_classify_micro(spec, org)
            result['fallback'] = 'regex'

        gemma_micro_map[combo] = result
        if (i + 1) % 50 == 0:
            with open(GEMMA_MICRO_MAP_FILE, 'w') as f:
                json.dump(gemma_micro_map, f, indent=2)

    with open(GEMMA_MICRO_MAP_FILE, 'w') as f:
        json.dump(gemma_micro_map, f, indent=2)

    # Summary
    dcounts = defaultdict(int)
    for v in gemma_drug_map.values():
        dcounts[v] += 1
    log(f"  Gemma drugs: {dict(dcounts)}")

    del llm
    gc.collect()
    return gemma_drug_map, gemma_micro_map


# ==========================================
# STEP 4: VALIDATION
# ==========================================

def run_validation(regex_drug, gemma_drug, regex_micro, gemma_micro):
    log("\nSTEP 4: Validation...")

    if gemma_drug is None:
        log("  Skipping (no MedGemma results)")
        return

    report = {
        'drugs': {'total': 0, 'agree': 0, 'disagree': 0, 'agreement_rate': 0.0,
                  'gemma_only_catches': [], 'regex_only_catches': [],
                  'disagreements': [], 'category_matrix': {}},
        'micro': {'total': 0, 'agree': 0, 'disagree': 0, 'agreement_rate': 0.0,
                  'bc_agree': 0, 'bc_disagree': 0, 'status_agree': 0, 'status_disagree': 0,
                  'disagreements': []}
    }

    # Drugs
    common = set(regex_drug.keys()) & set(gemma_drug.keys())
    report['drugs']['total'] = len(common)

    # Confusion matrix
    matrix = defaultdict(lambda: defaultdict(int))

    for drug in common:
        r, g = regex_drug[drug], gemma_drug[drug]
        matrix[r][g] += 1
        if r == g:
            report['drugs']['agree'] += 1
        else:
            report['drugs']['disagree'] += 1
            entry = {'drug': drug, 'regex': r, 'gemma': g}
            report['drugs']['disagreements'].append(entry)
            if r == 'other' and g in ('antibiotic', 'vasopressor', 'iv_fluid'):
                report['drugs']['gemma_only_catches'].append(entry)
            if g == 'other' and r in ('antibiotic', 'vasopressor', 'iv_fluid'):
                report['drugs']['regex_only_catches'].append(entry)

    report['drugs']['category_matrix'] = {k: dict(v) for k, v in matrix.items()}
    if report['drugs']['total'] > 0:
        report['drugs']['agreement_rate'] = round(
            report['drugs']['agree'] / report['drugs']['total'] * 100, 2)

    # Micro
    if gemma_micro:
        common_m = set(regex_micro.keys()) & set(gemma_micro.keys())
        report['micro']['total'] = len(common_m)

        for combo in common_m:
            r, g = regex_micro[combo], gemma_micro[combo]
            r_bc = r.get('is_blood_culture', False)
            g_bc = g.get('is_blood_culture', False)
            r_st = r.get('status', 'no_growth')
            g_st = g.get('status', 'no_growth')

            if r_bc == g_bc:
                report['micro']['bc_agree'] += 1
            else:
                report['micro']['bc_disagree'] += 1
            if r_st == g_st:
                report['micro']['status_agree'] += 1
            else:
                report['micro']['status_disagree'] += 1

            if r_bc == g_bc and r_st == g_st:
                report['micro']['agree'] += 1
            else:
                report['micro']['disagree'] += 1
                report['micro']['disagreements'].append({
                    'combo': combo, 'regex': r, 'gemma': g})

        if report['micro']['total'] > 0:
            report['micro']['agreement_rate'] = round(
                report['micro']['agree'] / report['micro']['total'] * 100, 2)

    with open(VALIDATION_FILE, 'w') as f:
        json.dump(report, f, indent=2)

    log(f"  Drug agreement: {report['drugs']['agreement_rate']}% "
        f"({report['drugs']['agree']}/{report['drugs']['total']})")
    log(f"  Gemma-only: {len(report['drugs']['gemma_only_catches'])}")
    log(f"  Regex-only: {len(report['drugs']['regex_only_catches'])}")
    log(f"  Micro overall: {report['micro']['agreement_rate']}%")
    log(f"  Micro BC agree: {report['micro']['bc_agree']}, "
        f"Status agree: {report['micro']['status_agree']}")


# ==========================================
# STEP 5: BUILD PER-EPISODE JSON
# ==========================================

def build_episode_json(ep_path, drug_map, micro_map, source_label):
    """Build normalized_events JSON for one episode."""
    profile_path = os.path.join(ep_path, 'static_profile.json')
    if not os.path.exists(profile_path):
        return None

    with open(profile_path, 'r') as f:
        profile = json.load(f)

    icu_stays = profile.get('icu_stays', [])
    t0 = icu_stays[0]['intime'] if icu_stays else profile.get('admittime')
    stay_id = icu_stays[0]['stay_id'] if icu_stays else None

    episode = {
        'meta': {
            'subject_id': profile['subject_id'],
            'hadm_id': profile['hadm_id'],
            'stay_id': stay_id,
            'gender': profile.get('gender'),
            'age': profile.get('anchor_age'),
            'admittime': profile.get('admittime'),
            'dischtime': profile.get('dischtime'),
            'icu_intime': icu_stays[0]['intime'] if icu_stays else None,
            'icu_outtime': icu_stays[0]['outtime'] if icu_stays else None,
            't0': t0,
            'source': source_label
        },
        'blood_cultures': [],
        'antibiotics': [],
        'lactate': [],
        'fluids': [],
        'vasopressors': [],
        'vitals': []
    }

    # --- Blood Cultures ---
    micro_path = os.path.join(ep_path, 'microbiology.csv')
    if os.path.exists(micro_path):
        try:
            df = pd.read_csv(micro_path)
            for _, row in df.iterrows():
                spec = str(row.get('spec_type_desc', 'UNKNOWN'))
                org = str(row.get('org_name', 'UNKNOWN'))
                if spec == 'nan': spec = 'UNKNOWN'
                if org == 'nan': org = 'UNKNOWN'
                key = f"{spec}|{org}"
                cls = micro_map.get(key, {'is_blood_culture': False, 'status': 'no_growth'})
                episode['blood_cultures'].append({
                    'charttime': str(row.get('charttime', '')),
                    'specimen': spec,
                    'organism': org,
                    'is_blood_culture': cls.get('is_blood_culture', False),
                    'pathogen_status': cls.get('status', 'no_growth'),
                    'source': source_label
                })
        except Exception:
            pass

    # --- Meds (admin + ordered) ---
    for file, col in [('medications_admin.csv', 'medication'),
                       ('medications_ordered.csv', 'drug')]:
        fpath = os.path.join(ep_path, file)
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath)
            if col not in df.columns:
                continue
            for _, row in df.iterrows():
                drug_name = str(row.get(col, ''))
                if drug_name in ('nan', 'None', ''):
                    continue
                category = drug_map.get(drug_name, 'other')
                ts = str(row.get('charttime', row.get('starttime', '')))
                if ts == 'nan': ts = ''

                if category == 'antibiotic':
                    episode['antibiotics'].append({
                        'charttime': ts, 'drug': drug_name,
                        'category': 'antibiotic',
                        'source_file': file, 'source': source_label
                    })
                elif category == 'vasopressor':
                    episode['vasopressors'].append({
                        'starttime': ts,
                        'endtime': str(row.get('stoptime', row.get('endtime', ''))),
                        'drug': drug_name, 'rate': None, 'rate_unit': None,
                        'category': 'vasopressor',
                        'source_file': file, 'source': source_label
                    })
                elif category == 'iv_fluid':
                    episode['fluids'].append({
                        'starttime': ts,
                        'endtime': str(row.get('stoptime', row.get('endtime', ''))),
                        'drug': drug_name, 'amount_ml': None,
                        'category': 'iv_fluid', 'patient_weight_kg': None,
                        'source_file': file, 'source': source_label
                    })
        except Exception:
            pass

    # --- Inputs (fluids/vasopressors with amounts) ---
    inputs_path = os.path.join(ep_path, 'inputs.csv')
    if os.path.exists(inputs_path):
        try:
            df = pd.read_csv(inputs_path)
            for _, row in df.iterrows():
                ordcat = str(row.get('ordercategoryname', ''))
                input_cat = regex_classify_input(ordcat)
                weight = float(row['patientweight']) if pd.notna(row.get('patientweight')) else None
                amount = float(row['amount']) if pd.notna(row.get('amount')) else None
                rate = float(row['rate']) if pd.notna(row.get('rate')) else None

                if input_cat == 'iv_fluid':
                    episode['fluids'].append({
                        'starttime': str(row.get('starttime', '')),
                        'endtime': str(row.get('endtime', '')),
                        'drug': f"itemid_{row.get('itemid', '')}",
                        'amount_ml': amount,
                        'amount_unit': str(row.get('amountuom', '')),
                        'category': 'iv_fluid',
                        'patient_weight_kg': weight,
                        'source_file': 'inputs.csv', 'source': source_label
                    })
                elif input_cat == 'antibiotic':
                    episode['antibiotics'].append({
                        'charttime': str(row.get('starttime', '')),
                        'drug': f"itemid_{row.get('itemid', '')}",
                        'category': 'antibiotic',
                        'source_file': 'inputs.csv', 'source': source_label
                    })
                elif input_cat == 'drip':
                    episode['vasopressors'].append({
                        'starttime': str(row.get('starttime', '')),
                        'endtime': str(row.get('endtime', '')),
                        'drug': f"itemid_{row.get('itemid', '')}",
                        'rate': rate,
                        'rate_unit': str(row.get('rateuom', '')),
                        'category': 'drip_possible_vasopressor',
                        'patient_weight_kg': weight,
                        'source_file': 'inputs.csv', 'source': source_label
                    })
        except Exception:
            pass

    # --- Lactate ---
    labs_path = os.path.join(ep_path, 'labs.csv')
    if os.path.exists(labs_path):
        try:
            df = pd.read_csv(labs_path)
            for _, row in df[df['itemid'].isin(LACTATE_ITEMIDS)].iterrows():
                if pd.notna(row.get('valuenum')):
                    episode['lactate'].append({
                        'charttime': str(row.get('charttime', '')),
                        'value': float(row['valuenum']),
                        'unit': 'mmol/L'
                    })
        except Exception:
            pass

    # --- Vitals ---
    vitals_path = os.path.join(ep_path, 'vitals.csv')
    if os.path.exists(vitals_path):
        try:
            df = pd.read_csv(vitals_path)
            df = df[df['itemid'].isin(VITAL_ITEMIDS.keys())].dropna(subset=['valuenum'])
            df['vital_type'] = df['itemid'].map(VITAL_ITEMIDS)
            for ct, grp in df.groupby('charttime'):
                vrow = {'charttime': str(ct)}
                for _, r in grp.iterrows():
                    vrow[r['vital_type']] = float(r['valuenum'])
                episode['vitals'].append(vrow)
            episode['vitals'].sort(key=lambda x: x.get('charttime', ''))
        except Exception:
            pass

    # Sort
    for k in ['blood_cultures', 'antibiotics', 'lactate']:
        episode[k].sort(key=lambda x: x.get('charttime', ''))
    for k in ['fluids', 'vasopressors']:
        episode[k].sort(key=lambda x: x.get('starttime', ''))

    return episode


def build_all_episodes(regex_drug, regex_micro, gemma_drug=None, gemma_micro=None):
    log("\nSTEP 5: Building per-episode normalized JSONs...")

    subjects = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    stats = {'total': 0, 'regex': 0, 'gemma': 0}

    for subj_id in tqdm(subjects, desc="Building"):
        subj_path = os.path.join(DATA_DIR, subj_id)
        for hadm_id in os.listdir(subj_path):
            ep_path = os.path.join(subj_path, hadm_id)
            if not os.path.isdir(ep_path):
                continue
            stats['total'] += 1

            ep = build_episode_json(ep_path, regex_drug, regex_micro, 'regex')
            if ep:
                with open(os.path.join(ep_path, 'normalized_events_regex.json'), 'w') as f:
                    json.dump(ep, f, indent=2)
                stats['regex'] += 1

            if gemma_drug and gemma_micro:
                ep = build_episode_json(ep_path, gemma_drug, gemma_micro, 'gemma')
                if ep:
                    with open(os.path.join(ep_path, 'normalized_events_gemma.json'), 'w') as f:
                        json.dump(ep, f, indent=2)
                    stats['gemma'] += 1

    log(f"  Episodes: {stats['total']}, Regex: {stats['regex']}, Gemma: {stats['gemma']}")


# ==========================================
# MAIN
# ==========================================

def main():
    t0 = time.time()
    log("=" * 60)
    log("NeSy SEPSIS PIPELINE")
    log("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    drugs, micro = extract_unique_strings()
    regex_drug, regex_micro = run_regex_baseline(drugs, micro)
    gemma_drug, gemma_micro = run_medgemma(drugs, micro)
    run_validation(regex_drug, gemma_drug, regex_micro, gemma_micro)
    build_all_episodes(regex_drug, regex_micro, gemma_drug, gemma_micro)

    log(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()