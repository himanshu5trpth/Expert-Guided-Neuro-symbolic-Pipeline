r"""
=============================================================================
MIMIC-IV Sepsis Cohort Extractor
=============================================================================
Handles the MIMIC-IV download structure where EVERY table is a folder:
  - hosp/emar.csv/          → contains emar.csv.gz (single file inside)
  - hosp/diagnoses_icd.csv/ → contains 279,585 0KB files (filename = row)
  - icu/chartevents.csv/    → contains chartevents.csv.gz (single file)

The folder is always named "<tablename>.csv" and inside is either:
  (a) A single .csv.gz file with the actual data, OR
  (b) Thousands of 0KB files whose filenames ARE the CSV data rows

Usage:
  python extractor.py --source_dir ./data/mimic-iv --output_dir ./output
=============================================================================
"""

import pandas as pd
import os
import time
import gc
import sys

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_DIR = r"./mimic-iv"
OUTPUT_DIR = r"./septic_mimic_output"

ICD9_PREFIXES  = ["038"]
ICD9_CODES     = ["99591", "99592", "78552"]
ICD10_PREFIXES = ["A40", "A41"]
ICD10_CODES    = ["R6520", "R6521"]

COMPRESS_OUTPUT = True


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ==========================================
# SCHEMAS FOR FOLDER-AS-FILENAMES TABLES
# ==========================================
FOLDER_SCHEMAS = {
    'diagnoses_icd':  ['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version'],
    'procedures_icd': ['subject_id', 'hadm_id', 'seq_num', 'chartdate', 'icd_code', 'icd_version'],
    'provider':       ['provider_id'],
}

NUMERIC_COLS = {
    'subject_id', 'hadm_id', 'stay_id', 'transfer_id', 'seq_num',
    'itemid', 'microevent_id', 'spec_itemid', 'test_itemid',
    'org_itemid', 'ab_itemid', 'test_seq', 'isolate_num',
    'caregiver_id', 'micro_specimen_id', 'anchor_age', 'anchor_year',
    'hospital_expire_flag', 'icd_version', 'drg_severity', 'drg_mortality',
    'warning', 'labevent_id', 'specimen_id', 'emar_seq',
    'parent_field_ordinal', 'poe_seq',
}


# ==========================================
# FILE RESOLUTION
# ==========================================

def resolve_table(folder, table_name):
    """
    Finds the actual data source for a table.

    Structure: SOURCE_DIR/folder/table_name.csv/ (a DIRECTORY)
      Inside: either a single .csv.gz or many 0KB filename-as-data files

    Returns: (path, format_type) or (None, None)
    """
    base = os.path.join(SOURCE_DIR, folder)

    # Primary: table_name.csv is a FOLDER
    table_dir = os.path.join(base, f"{table_name}.csv")
    if os.path.isdir(table_dir):
        contents = os.listdir(table_dir)
        if not contents:
            return None, None

        gz_files = [f for f in contents if f.endswith('.csv.gz')]
        if gz_files:
            return os.path.join(table_dir, gz_files[0]), 'csv_gz'

        csv_files = [f for f in contents if f.endswith('.csv')]
        if csv_files:
            return os.path.join(table_dir, csv_files[0]), 'csv'

        if len(contents) > 10:
            return table_dir, 'folder_filenames'

        if len(contents) == 1:
            return os.path.join(table_dir, contents[0]), 'csv_gz'

        return None, None

    # Fallback: direct file
    gz_path = os.path.join(base, f"{table_name}.csv.gz")
    if os.path.isfile(gz_path):
        return gz_path, 'csv_gz'

    csv_path = os.path.join(base, f"{table_name}.csv")
    if os.path.isfile(csv_path):
        return csv_path, 'csv'

    return None, None


# ==========================================
# READERS
# ==========================================

def read_folder_as_filenames(dir_path, table_name):
    """Reconstructs DataFrame from folder of 0KB files."""
    if table_name not in FOLDER_SCHEMAS:
        files = os.listdir(dir_path)
        if not files:
            return pd.DataFrame()
        n_cols = len(files[0].split(','))
        columns = [f"col_{i}" for i in range(n_cols)]
        log(f"    Unknown schema for '{table_name}', guessed {n_cols} cols")
    else:
        columns = FOLDER_SCHEMAS[table_name]

    rows = []
    for fname in os.listdir(dir_path):
        parts = fname.split(',')
        if len(parts) < len(columns):
            parts.extend([''] * (len(columns) - len(parts)))
        elif len(parts) > len(columns):
            parts = parts[:len(columns)]
        rows.append(parts)

    df = pd.DataFrame(rows, columns=columns)
    for col in df.columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace('', pd.NA)
        if col in NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def read_table(folder, table_name):
    """Reads a table, returns full DataFrame."""
    filepath, fmt = resolve_table(folder, table_name)
    if filepath is None:
        log(f"  WARNING: {folder}/{table_name} not found. Skipping.")
        return None

    if fmt in ('csv_gz', 'csv'):
        log(f"  Reading {folder}/{table_name} [{fmt}] ...")
        return pd.read_csv(filepath, low_memory=False)
    elif fmt == 'folder_filenames':
        log(f"  Reading {folder}/{table_name} [folder-as-filenames] ...")
        df = read_folder_as_filenames(filepath, table_name)
        log(f"    Reconstructed {len(df):,} rows")
        return df
    return None


def read_table_chunked(folder, table_name, chunksize=500_000):
    """Returns (source, mode) — 'chunked' iterator or 'full' DataFrame."""
    filepath, fmt = resolve_table(folder, table_name)
    if filepath is None:
        log(f"  WARNING: {folder}/{table_name} not found. Skipping.")
        return None, None

    if fmt in ('csv_gz', 'csv'):
        log(f"  Reading {folder}/{table_name} [{fmt}] (chunked)...")
        try:
            return pd.read_csv(filepath, chunksize=chunksize, low_memory=False), 'chunked'
        except Exception as e:
            log(f"    Chunked failed ({e}), trying full...")
            try:
                return pd.read_csv(filepath, low_memory=False), 'full'
            except Exception as e2:
                log(f"    Full also failed: {e2}")
                return None, None
    elif fmt == 'folder_filenames':
        log(f"  Reading {folder}/{table_name} [folder-as-filenames] (full)...")
        df = read_folder_as_filenames(filepath, table_name)
        log(f"    Reconstructed {len(df):,} rows")
        return df, 'full'
    return None, None


# ==========================================
# FILTER & SAVE
# ==========================================

def save_table(df, folder, filename):
    out_folder = os.path.join(OUTPUT_DIR, folder)
    os.makedirs(out_folder, exist_ok=True)
    ext = '.csv.gz' if COMPRESS_OUTPUT else '.csv'
    out_path = os.path.join(out_folder, f"{filename}{ext}")
    df.to_csv(out_path, index=False,
              compression='gzip' if COMPRESS_OUTPUT else None)
    log(f"  Saved {folder}/{filename}{ext} ({len(df):,} rows)")


def _build_mask(df, filter_col, filter_ids, alt_col=None, alt_ids=None):
    if filter_col in df.columns:
        col = pd.to_numeric(df[filter_col], errors='coerce')
        mask = col.isin(filter_ids)
    elif 'subject_id' in df.columns:
        col = pd.to_numeric(df['subject_id'], errors='coerce')
        mask = col.isin(alt_ids if alt_ids else filter_ids)
    else:
        return pd.Series([False] * len(df), index=df.index)

    if alt_col and alt_ids and filter_col in df.columns and alt_col in df.columns:
        primary_null = pd.to_numeric(df[filter_col], errors='coerce').isna()
        alt_data = pd.to_numeric(df[alt_col], errors='coerce')
        mask = mask | (primary_null & alt_data.isin(alt_ids))

    return mask


def filter_large_table(folder, table_name, filter_col, filter_ids,
                        alt_col=None, alt_ids=None):
    source, mode = read_table_chunked(folder, table_name)
    if source is None:
        return

    if mode == 'full':
        mask = _build_mask(source, filter_col, filter_ids, alt_col, alt_ids)
        filtered = source[mask]
        if len(filtered) > 0:
            save_table(filtered, folder, table_name)
        else:
            log(f"  No matching rows for {table_name}")
        del source, filtered
        gc.collect()
        return

    # Chunked
    all_chunks = []
    total_in = 0
    total_out = 0
    try:
        for i, chunk in enumerate(source):
            total_in += len(chunk)
            mask = _build_mask(chunk, filter_col, filter_ids, alt_col, alt_ids)
            filtered = chunk[mask]
            total_out += len(filtered)
            if len(filtered) > 0:
                all_chunks.append(filtered.copy())
            if (i + 1) % 10 == 0:
                log(f"    ... {total_in:,} rows in, {total_out:,} kept")
            del chunk, filtered
            gc.collect()
    except Exception as e:
        log(f"    ERROR during chunked read: {e}")

    if all_chunks:
        combined = pd.concat(all_chunks, ignore_index=True)
        save_table(combined, folder, table_name)
        del combined
    else:
        log(f"  No matching rows for {table_name}")

    del all_chunks
    gc.collect()
    log(f"  Done: {folder}/{table_name} → {total_out:,}/{total_in:,} rows")


def is_sepsis_code(icd_code, icd_version):
    code = str(icd_code).strip()
    try:
        version = int(float(icd_version)) if pd.notnull(icd_version) else 0
    except (ValueError, TypeError):
        version = 0
    if version == 9:
        return code in ICD9_CODES or any(code.startswith(p) for p in ICD9_PREFIXES)
    elif version == 10:
        return code in ICD10_CODES or any(code.startswith(p) for p in ICD10_PREFIXES)
    return False


# ==========================================
# STEP 1: IDENTIFY SEPSIS PATIENTS
# ==========================================

def identify_sepsis_cohort():
    log("=" * 60)
    log("STEP 1: Identifying sepsis patients")
    log("=" * 60)

    diag = read_table('hosp', 'diagnoses_icd')
    if diag is None or diag.empty:
        log("FATAL: Cannot read diagnoses_icd.")
        sys.exit(1)

    log(f"  Total diagnosis rows: {len(diag):,}")

    diag['icd_code'] = diag['icd_code'].astype(str).str.strip()
    diag['icd_version'] = pd.to_numeric(diag['icd_version'], errors='coerce')
    diag['hadm_id'] = pd.to_numeric(diag['hadm_id'], errors='coerce')
    diag['subject_id'] = pd.to_numeric(diag['subject_id'], errors='coerce')

    sepsis_mask = diag.apply(
        lambda row: is_sepsis_code(row['icd_code'], row['icd_version']), axis=1)
    sepsis_diag = diag[sepsis_mask]
    log(f"  Sepsis diagnosis rows: {len(sepsis_diag):,}")

    sepsis_hadm = set(sepsis_diag['hadm_id'].dropna().astype(int).unique())
    sepsis_subj = set(sepsis_diag['subject_id'].dropna().astype(int).unique())
    log(f"  Unique hadm_ids: {len(sepsis_hadm):,}")
    log(f"  Unique subject_ids: {len(sepsis_subj):,}")

    log("\n  Code distribution:")
    counts = sepsis_diag.groupby(['icd_version', 'icd_code']).size()
    for (ver, code), n in counts.items():
        log(f"    ICD-{int(ver)} {code}: {n:,}")

    return sepsis_subj, sepsis_hadm, diag


# ==========================================
# STEP 2: FILTER HOSP TABLES
# ==========================================

def filter_hosp(subj_ids, hadm_ids, diag):
    log("\n" + "=" * 60)
    log("STEP 2: Filtering hosp/ tables")
    log("=" * 60)

    # diagnoses_icd: ALL diagnoses for sepsis hadm_ids
    log("\n[hosp/diagnoses_icd]")
    save_table(diag[diag['hadm_id'].isin(hadm_ids)], 'hosp', 'diagnoses_icd')
    del diag
    gc.collect()

    # Small tables by hadm_id
    for tname in ['admissions', 'drgcodes', 'hcpcsevents', 'services',
                   'microbiologyevents', 'procedures_icd']:
        log(f"\n[hosp/{tname}]")
        df = read_table('hosp', tname)
        if df is None or df.empty:
            continue
        if 'hadm_id' in df.columns:
            df['hadm_id'] = pd.to_numeric(df['hadm_id'], errors='coerce')
            save_table(df[df['hadm_id'].isin(hadm_ids)], 'hosp', tname)
        del df
        gc.collect()

    # transfers (hadm_id + subject_id fallback)
    log("\n[hosp/transfers]")
    df = read_table('hosp', 'transfers')
    if df is not None and not df.empty:
        df['hadm_id'] = pd.to_numeric(df['hadm_id'], errors='coerce')
        df['subject_id'] = pd.to_numeric(df['subject_id'], errors='coerce')
        mask = df['hadm_id'].isin(hadm_ids) | (
            df['hadm_id'].isna() & df['subject_id'].isin(subj_ids))
        save_table(df[mask], 'hosp', 'transfers')
        del df
        gc.collect()

    # patients (by subject_id)
    log("\n[hosp/patients]")
    df = read_table('hosp', 'patients')
    if df is not None and not df.empty:
        df['subject_id'] = pd.to_numeric(df['subject_id'], errors='coerce')
        save_table(df[df['subject_id'].isin(subj_ids)], 'hosp', 'patients')
        del df
        gc.collect()

    # Large tables (chunked)
    large = [
        ('labevents',     'hadm_id',    hadm_ids, 'subject_id', subj_ids),
        ('emar',          'hadm_id',    hadm_ids, 'subject_id', subj_ids),
        ('emar_detail',   'subject_id', subj_ids, None,         None),
        ('prescriptions', 'hadm_id',    hadm_ids, None,         None),
        ('pharmacy',      'hadm_id',    hadm_ids, None,         None),
        ('poe',           'hadm_id',    hadm_ids, None,         None),
        ('poe_detail',    'subject_id', subj_ids, None,         None),
        ('omr',           'subject_id', subj_ids, None,         None),
    ]
    for tname, fcol, fids, acol, aids in large:
        log(f"\n[hosp/{tname}]")
        filter_large_table('hosp', tname, fcol, fids, acol, aids)

    # Dictionary tables (copy as-is)
    log("\n[hosp/dictionaries - copy as-is]")
    for tname in ['d_hcpcs', 'd_icd_diagnoses', 'd_icd_procedures',
                   'd_labitems', 'provider']:
        df = read_table('hosp', tname)
        if df is not None and not df.empty:
            save_table(df, 'hosp', tname)
            del df
    gc.collect()


# ==========================================
# STEP 3: FILTER ICU TABLES
# ==========================================

def filter_icu(subj_ids, hadm_ids):
    log("\n" + "=" * 60)
    log("STEP 3: Filtering icu/ tables")
    log("=" * 60)

    log("\n[icu/icustays]")
    icu = read_table('icu', 'icustays')
    if icu is None or icu.empty:
        log("WARNING: icustays not found. Skipping ICU.")
        return set()

    icu['hadm_id'] = pd.to_numeric(icu['hadm_id'], errors='coerce')
    icu['stay_id'] = pd.to_numeric(icu['stay_id'], errors='coerce')
    filtered = icu[icu['hadm_id'].isin(hadm_ids)]
    stay_ids = set(filtered['stay_id'].dropna().astype(int).unique())
    save_table(filtered, 'icu', 'icustays')
    log(f"  Sepsis ICU stays: {len(stay_ids):,}")
    del icu, filtered
    gc.collect()

    if not stay_ids:
        log("  No ICU stays. Skipping remaining ICU tables.")
        return stay_ids

    # Large ICU tables
    for tname in ['chartevents', 'inputevents', 'outputevents',
                   'datetimeevents', 'ingredientevents']:
        log(f"\n[icu/{tname}]")
        filter_large_table('icu', tname, 'stay_id', stay_ids)

    # procedureevents (smaller)
    log("\n[icu/procedureevents]")
    df = read_table('icu', 'procedureevents')
    if df is not None and not df.empty:
        df['stay_id'] = pd.to_numeric(df['stay_id'], errors='coerce')
        save_table(df[df['stay_id'].isin(stay_ids)], 'icu', 'procedureevents')
        del df
        gc.collect()

    # Dictionaries
    log("\n[icu/dictionaries - copy as-is]")
    for tname in ['d_items', 'caregiver']:
        df = read_table('icu', tname)
        if df is not None and not df.empty:
            save_table(df, 'icu', tname)
            del df
    gc.collect()
    return stay_ids


# ==========================================
# STEP 4: SUMMARY
# ==========================================

def generate_summary(subj_ids, hadm_ids, stay_ids):
    log("\n" + "=" * 60)
    log("STEP 4: Summary")
    log("=" * 60)

    total_size = 0
    file_count = 0
    details = []
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp)
            total_size += sz
            file_count += 1
            details.append((os.path.relpath(fp, OUTPUT_DIR), sz))

    with open(os.path.join(OUTPUT_DIR, 'cohort_summary.txt'), 'w') as f:
        f.write("MIMIC-IV Sepsis Cohort Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Source: {SOURCE_DIR}\n")
        f.write(f"Output: {OUTPUT_DIR}\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Patients:          {len(subj_ids):,}\n")
        f.write(f"Hospitalizations:  {len(hadm_ids):,}\n")
        f.write(f"ICU stays:         {len(stay_ids):,}\n")
        f.write(f"Output files:      {file_count}\n")
        f.write(f"Total size:        {total_size/(1024**3):.2f} GB\n\n")
        f.write("Files:\n")
        for path, sz in sorted(details):
            f.write(f"  {path:50s} {sz/1024/1024:>8.1f} MB\n")

    pd.DataFrame({'subject_id': list(subj_ids)}).to_csv(
        os.path.join(OUTPUT_DIR, 'sepsis_subject_ids.csv'), index=False)
    pd.DataFrame({'hadm_id': list(hadm_ids)}).to_csv(
        os.path.join(OUTPUT_DIR, 'sepsis_hadm_ids.csv'), index=False)
    pd.DataFrame({'stay_id': list(stay_ids)}).to_csv(
        os.path.join(OUTPUT_DIR, 'sepsis_stay_ids.csv'), index=False)

    log(f"  Patients:         {len(subj_ids):,}")
    log(f"  Hospitalizations: {len(hadm_ids):,}")
    log(f"  ICU stays:        {len(stay_ids):,}")
    log(f"  Output files:     {file_count}")
    log(f"  Total size:       {total_size/(1024**3):.2f} GB")


# ==========================================
# MAIN
# ==========================================

def main():
    t0 = time.time()

    log("=" * 60)
    log("MIMIC-IV SEPSIS COHORT EXTRACTOR")
    log("=" * 60)
    log(f"Source: {SOURCE_DIR}")
    log(f"Output: {OUTPUT_DIR}")

    if not os.path.exists(SOURCE_DIR):
        log(f"FATAL: {SOURCE_DIR} not found.")
        sys.exit(1)

    # Quick scan to confirm structure
    log("\nScanning structure...")
    for sub in ['hosp', 'icu']:
        sub_path = os.path.join(SOURCE_DIR, sub)
        if not os.path.exists(sub_path):
            log(f"  {sub}/ NOT FOUND")
            continue
        items = sorted(os.listdir(sub_path))
        log(f"  {sub}/: {len(items)} items")
        for item in items:
            item_path = os.path.join(sub_path, item)
            if os.path.isdir(item_path):
                contents = os.listdir(item_path)
                if len(contents) == 1:
                    inner = contents[0]
                    inner_size = os.path.getsize(os.path.join(item_path, inner))
                    log(f"    {item:40s} → {inner} ({inner_size/1024/1024:.1f} MB)")
                elif len(contents) > 1:
                    log(f"    {item:40s} → {len(contents):,} filename-rows")
                else:
                    log(f"    {item:40s} → EMPTY")
            else:
                sz = os.path.getsize(item_path)
                log(f"    {item:40s} → file ({sz/1024/1024:.1f} MB)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subj_ids, hadm_ids, diag = identify_sepsis_cohort()
    if not hadm_ids:
        log("FATAL: No sepsis patients found.")
        sys.exit(1)

    filter_hosp(subj_ids, hadm_ids, diag)
    del diag
    gc.collect()

    stay_ids = filter_icu(subj_ids, hadm_ids)
    generate_summary(subj_ids, hadm_ids, stay_ids)

    log(f"\nTotal time: {(time.time()-t0)/60:.1f} minutes")
    log(f"Output: {OUTPUT_DIR}")
    log("Done!")


if __name__ == "__main__":
    main()