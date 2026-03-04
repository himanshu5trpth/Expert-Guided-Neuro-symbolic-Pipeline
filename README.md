# Expert-Guided Neuro-Symbolic Pipeline for Sepsis Compliance Assessment

> **This is a clinical insight generation pipeline, not an automated decision-making system.** Every compliance score produced by this pipeline is intended to inform healthcare professionals — never to replace clinical judgment.

![Pipeline Overview](figures/fig1_pipeline_overview.png)

## What This Pipeline Does

This pipeline evaluates how well sepsis treatment in hospital records aligns with the [Surviving Sepsis Campaign (SSC)](https://sccm.org/survivingsepsiscampaign/guidelines-and-resources/surviving-sepsis-campaign-adult-guidelines) bundle guidelines. It processes raw Electronic Health Records from the [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) database through six stages — cohort extraction, data structuring, sampling, semantic normalization, embedding validation, and fuzzy compliance scoring — to produce graded compliance insights across eight clinical rules.

The key contribution is the **hybrid neuro-symbolic approach**: an LLM (MedGemma) handles semantic normalization of messy clinical text (trade names, abbreviations, tall-man lettering), while a Sugeno fuzzy inference system with expert-validated parameters produces interpretable, graded compliance scores rather than brittle binary pass/fail judgments. Subject matter experts (SMEs) are consulted at critical decision points throughout the pipeline.

### Eight SSC Bundle Rules Evaluated

| Phase | Rule | Clinical Target |
|-------|------|-----------------|
| **Hour-1 Bundle** | R1 | Blood cultures obtained before antibiotics |
| | R2 | Broad-spectrum antibiotics within 1 hour |
| | R3 | Lactate measured within 1 hour *(highest priority)* |
| | R4† | Repeat lactate within 2–4 hours if initial ≥ 2.0 mmol/L |
| **Resuscitation** | R5† | 30 mL/kg IV fluids if MAP < 65 or lactate ≥ 4 |
| | R6† | Vasopressors if MAP < 65 after fluids |
| **Response** | R7† | MAP recovery ≥ 65 mmHg post-vasopressors |
| | R8† | Lactate clearance ≥ 10% |

*† Conditional rules — activate only when clinical triggers are met.*

---

## Repository Structure

```
NeSy-Sepsis-Pipeline/
│
├── README.md
├── requirements.txt
│
├── figures/
│   ├── fig1_pipeline_overview.png       # End-to-end workflow diagram
│   └── fig3_pipeline_architecture.png   # Detailed pipeline with numbers
│
├── pipeline/                            # Core pipeline scripts (run in order)
│   ├── 1_Crop_Mimic.py                  # Stage 1: Cohort extraction
│   ├── 2_Group_Patient.py               # Stage 2: Episode-centric restructuring
│   ├── 3_Sample_Random.py               # Stage 3: Reproducible random sampling
│   ├── 4_Semantic_Normalization.py       # Stage 4: Regex + MedGemma classification
│   ├── 5_Embedding_Check.py             # Stage 5: Embedding validation & adjudication
│   └── 6_Run_Fuzzy.py                   # Stage 6: Sugeno fuzzy compliance scoring
│
├── output/                              # Pre-computed pipeline outputs
│   ├── unique_strings.json              # 1,691 drug + 650 micro unique strings
│   ├── regex_drug_map.json              # Regex drug classifications
│   ├── regex_micro_map.json             # Regex microbiology classifications
│   ├── gemma_drug_map.json              # MedGemma drug classifications
│   ├── gemma_micro_map.json             # MedGemma microbiology classifications
│   ├── validation_report.json           # Inter-system agreement analysis
│   ├── embedding_validation_report.json # Embedding validation + adjudication
│   └── fuzzy_compliance_results.csv     # Final per-episode compliance scores
│
└── medgemma-4b-it-Q4_K_S.gguf          # MedGemma model (download separately)
```

---

## Pipeline Execution Order

The pipeline runs sequentially. Each stage produces outputs consumed by the next. **SME consultation points** are marked explicitly — these are where a domain expert reviews intermediate results before proceeding.

![Detailed Pipeline Architecture](figures/fig3_pipeline_architecture.png)

### Stage 1: Cohort Extraction — `1_Crop_Mimic.py`

Identifies all sepsis patients from raw MIMIC-IV using ICD-9/10 diagnosis codes and extracts only the relevant tables for this cohort.

| | |
|---|---|
| **Input** | `./mimic-iv/` — Raw MIMIC-IV v3.1 download (hosp/ and icu/ modules) |
| **Output** | `./Septic_Mimic/` — Filtered tables containing only sepsis-related records |
| **Key Logic** | ICD-9: `038.*`, `995.91`, `995.92`, `785.52`; ICD-10: `A40`, `A41`, `R65.20`, `R65.21` |

```bash
python pipeline/1_Crop_Mimic.py
```

**Result:** 17,926 unique sepsis patients across 22,363 hospitalizations identified and extracted.

---

### Stage 2: Episode-Centric Restructuring — `2_Group_Patient.py`

Transforms the relational MIMIC-IV structure into a patient/episode folder hierarchy. Each sepsis episode (hadm_id) gets its own folder with seven standardized files.

| | |
|---|---|
| **Input** | `./Septic_Mimic/` — Filtered MIMIC tables from Stage 1 |
| **Output** | `./Processed_Patients/{subject_id}/{hadm_id}/` — One folder per episode |

Each episode folder contains:
- `static_profile.json` — Demographics, admission/discharge times, ICU stays
- `medications_admin.csv` — eMAR administration records (Rule 2 timestamps)
- `medications_ordered.csv` — Prescriptions with routes (Rule 2 route filtering)
- `microbiology.csv` — Culture specimens and organisms (Rule 1)
- `labs.csv` — Lactate and laboratory values (Rules 3, 4, 8)
- `vitals.csv` — Blood pressure and vital signs (Rule 7)
- `inputs.csv` — IV fluids and vasopressors (Rules 5, 6)

```bash
python pipeline/2_Group_Patient.py
```

---

### Stage 3: Reproducible Random Sampling — `3_Sample_Random.py`

Draws a statistically powered random sample from the full cohort. The fixed seed ensures exact reproducibility.

| | |
|---|---|
| **Input** | `./Processed_Patients/` — Full structured cohort from Stage 2 |
| **Output** | `./2000_Patients_Sampled/` — Sampled subset (same folder structure) |
| **Parameters** | `SAMPLE_SIZE = 2000`, `RANDOM_SEED = 55` |

```bash
python pipeline/3_Sample_Random.py
```

**Result:** 2,000 patients → 2,438 sepsis episodes (some patients had multiple hospitalizations). The cohort exceeds minimum sample sizes for Cohen's Kappa (76.9×), McNemar's test (5.2×), and fuzzy membership estimation (10.2×).

---

### Stage 4: Semantic Normalization — `4_Semantic_Normalization.py`

The core neuro-symbolic stage. Extracts all unique drug and microbiology strings across the cohort, then classifies them using two complementary systems:

1. **Regex classifier** — Domain-informed pattern matching with exclusion rules
2. **MedGemma classifier** — Zero-shot structured prompting of MedGemma-4b-it (quantized, local execution)

| | |
|---|---|
| **Input** | `./2000_Patients_Sampled/` — Sampled episodes from Stage 3 |
| **Input** | `./medgemma-4b-it-Q4_K_S.gguf` — Quantized MedGemma model (root directory) |
| **Output** | `./NeSy_Output/unique_strings.json` — 1,691 drugs + 650 micro combinations |
| **Output** | `./NeSy_Output/regex_drug_map.json` — Regex: 141 antibiotics, 29 vasopressors, 50 IV fluids |
| **Output** | `./NeSy_Output/gemma_drug_map.json` — MedGemma: 158 antibiotics, 11 vasopressors, 44 IV fluids |
| **Output** | `./NeSy_Output/regex_micro_map.json` — Regex microbiology classifications |
| **Output** | `./NeSy_Output/gemma_micro_map.json` — MedGemma microbiology classifications |
| **Output** | `./NeSy_Output/validation_report.json` — Agreement: 94.26% drugs, 60.62% micro |
| **Output** | Per-episode `normalized_events_regex.json` and `normalized_events_gemma.json` |

```bash
python pipeline/4_Semantic_Normalization.py
```

**Requires:** `llama-cpp-python` with the MedGemma GGUF model file placed at the repository root.

---

### ⚕️ SME Consultation Point 1: Review Classification Outputs

**Before proceeding to Stage 5**, a subject matter expert should review:

- **`validation_report.json`** — The 97 drug disagreements and microbiology discrepancies
- **Category distributions** — Do the antibiotic/vasopressor/IV fluid counts look clinically reasonable?
- **Edge cases** — Are there drugs misclassified due to formulation context (e.g., Vancomycin Oral should not count as systemic antibiotic for SSC compliance)?

In our study, SME consultation at this stage produced these critical inputs:
1. Vancomycin Oral Liquid and Vancomycin Enema → reclassified as `other` (not systemic antibiotics)
2. Only IV/IM antibiotic routes count toward SSC compliance
3. Coagulase-negative Staphylococcus → `pathogen` in ICU populations (immunocompromised context)
4. Viridans streptococci in blood cultures → `pathogen`

---

### Stage 5: Embedding Validation & Adjudication — `5_Embedding_Check.py`

Uses MedGemma's embedding space as an independent semantic reference to validate classifications and resolve disagreements.

**Phase 1 — Validation:** The 1,594 agreed drug classifications serve as ground truth. Per-category cosine similarity thresholds are calibrated (mean − 2σ), and all 213 non-other MedGemma classifications are tested against them.

**Phase 2 — Adjudication:** Each of the 97 disagreed drugs is embedded and compared against both classifier-claimed category anchors. The side with higher cosine similarity wins.

| | |
|---|---|
| **Input** | `./NeSy_Output/regex_drug_map.json` — From Stage 4 |
| **Input** | `./NeSy_Output/gemma_drug_map.json` — From Stage 4 |
| **Input** | `./medgemma-4b-it-Q4_K_S.gguf` — MedGemma model for embedding extraction |
| **Output** | `./NeSy_Output/embedding_validation_report.json` |

```bash
python pipeline/5_Embedding_Check.py
```

**Result:** 210/213 confirmed (98.59%), 3 flagged (ALL-CAPS tokenizer artifacts). Adjudication: 49 regex wins, 48 MedGemma wins — confirming complementary error profiles.

---

### ⚕️ SME Consultation Point 2: Validate Before Fuzzy Scoring

**Before proceeding to Stage 6**, a subject matter expert should review:

- **Flagged drugs** from embedding validation — Are the 3 flags genuine misclassifications or tokenizer artifacts?
- **Adjudication results** — Do the 97 resolved disagreements look clinically sound?
- **Fuzzy parameters** — Review and approve the membership function parameters (σ values, rule weights, priority ordering) that will drive compliance scoring

In our study, SME consultation at this stage established:
1. Gaussian membership functions (reflecting gradual clinical transitions, not sharp boundaries)
2. Rule priority ordering: R3 > R2 > R5 > R6 > R1 = R4 = R7 = R8
3. σ = 30 minutes for timing-based Hour-1 rules (treatment at 90 min → µ = 0.61; at 120 min → µ = 0.13)

---

### Stage 6: Fuzzy Compliance Assessment — `6_Run_Fuzzy.py`

Builds a single consensus classification map (agreed + embedding-validated + SME overrides), then evaluates all 2,438 episodes through a Sugeno fuzzy inference system encoding eight SSC bundle rules.

| | |
|---|---|
| **Input** | `./2000_Patients_Sampled/` — Episode data from Stage 3 |
| **Input** | `./NeSy_Output/regex_drug_map.json` — From Stage 4 |
| **Input** | `./NeSy_Output/gemma_drug_map.json` — From Stage 4 |
| **Input** | `./NeSy_Output/regex_micro_map.json` — From Stage 4 |
| **Input** | `./NeSy_Output/gemma_micro_map.json` — From Stage 4 |
| **Input** | `./NeSy_Output/embedding_validation_report.json` — From Stage 5 |
| **Output** | `./NeSy_Output/fuzzy_compliance_results.csv` — One row per episode, 50 columns |

```bash
python pipeline/6_Run_Fuzzy.py
```

The output CSV contains per-episode: compliance score (0–1), all eight rule scores, raw measurements, applicability flags, missing data indicators, ICU length of stay, and worst-performing rule identification.

---

### ⚕️ SME Consultation Point 3: Interpret Results

**After Stage 6**, the compliance scores and clinical insights are presented to domain experts for interpretation and validation. This is where the pipeline delivers its value — the numbers become actionable only through clinical interpretation.

Key findings from our study that SMEs validated and contextualized:
- Antibiotic timing (R2, mean = 0.24) is the worst compliance area, but SMEs explained this partly reflects pre-ICU administration that the onset-anchored window penalizes
- High conditional-rule performance (R6 = 0.73, R7 = 0.97) reflects **survivorship bias**, not genuine compliance excellence
- The 51% patient drop-off at elevated lactate indicates unmeasured or undocumented lactate — both constituting bundle failures
- Higher compliance episodes achieve median ICU stays of 3.8 days vs 5.1 days for low compliance

---

## Pre-Computed Outputs

The `output/` directory contains all intermediate and final outputs from our pipeline run. These allow researchers to:

- **Skip Stages 1–3** if you don't have MIMIC-IV access (use the classification maps directly)
- **Skip Stage 4** if you don't have MedGemma (the pre-computed maps are provided)
- **Reproduce our analysis** by comparing your pipeline outputs against ours
- **Extend the work** by modifying fuzzy parameters or adding new rules

| File | Description | Records |
|------|-------------|---------|
| `unique_strings.json` | All unique drug and microbiology strings | 1,691 drugs + 650 micro |
| `regex_drug_map.json` | Regex classifications for all drugs | 1,691 entries |
| `gemma_drug_map.json` | MedGemma classifications for all drugs | 1,691 entries |
| `regex_micro_map.json` | Regex microbiology classifications | 650 entries |
| `gemma_micro_map.json` | MedGemma microbiology classifications | 650 entries |
| `validation_report.json` | Inter-system agreement + disagreement details | 97 drug disagreements |
| `embedding_validation_report.json` | Phase 1 validation + Phase 2 adjudication | 210 confirmed, 3 flagged |
| `fuzzy_compliance_results.csv` | Final per-episode compliance scores | 2,438 episodes |

---

## Requirements

```
pandas
numpy
tqdm
llama-cpp-python    # Only for Stages 4 and 5 (MedGemma inference)
```

**MedGemma Model:** Download `medgemma-4b-it-Q4_K_S.gguf` and place it in the repository root. This quantized model runs locally — no API calls, no data leaves your machine.

**MIMIC-IV Access:** Stages 1–3 require credentialed access to [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) from PhysioNet. The pre-computed outputs in `output/` allow downstream analysis without MIMIC access.

---

## Data Use and Privacy

This pipeline processes data from MIMIC-IV, which requires a signed Data Use Agreement with PhysioNet. The pre-computed output files contain derived aggregate statistics and classification maps — **no raw patient data, no identifiable information**. The `fuzzy_compliance_results.csv` contains MIMIC-IV `subject_id` and `hadm_id` identifiers; users must have their own MIMIC-IV credentials to use these in conjunction with the source database.

---

This code is released for academic and research purposes. The MIMIC-IV data it processes is governed by the [PhysioNet Credentialed Health Data Use Agreement](https://physionet.org/content/mimiciv/view-dua/3.1/).
