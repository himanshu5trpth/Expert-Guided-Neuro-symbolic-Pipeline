import pandas as pd
import os
import json
import gc
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Path to your "Septic_Mimic" folder from the previous step
SOURCE_DIR = r"./Septic_Mimic"
OUTPUT_DIR = r"./Processed_Patients"


def load_csv(folder, filename):
    """Safe loader that handles missing files and folder-based CSVs gracefully."""
    path_gz = os.path.join(SOURCE_DIR, folder, f"{filename}.csv.gz")
    path_csv = os.path.join(SOURCE_DIR, folder, f"{filename}.csv")

    # 1. Try GZIP file
    if os.path.exists(path_gz) and os.path.isfile(path_gz):
        print(f"Loading {filename} (gzip)...")
        return pd.read_csv(path_gz, low_memory=False)

    # 2. Try CSV (could be file or folder)
    elif os.path.exists(path_csv):
        # Case A: It is a directory (MIMIC "folder-as-file" structure)
        if os.path.isdir(path_csv):
            # Look for the actual csv file inside: e.g., patients.csv/patients.csv
            inner_path = os.path.join(path_csv, f"{filename}.csv")
            if os.path.exists(inner_path):
                print(f"Loading {filename} (from folder)...")
                return pd.read_csv(inner_path, low_memory=False)

            # Fallback: Just grab the first CSV found inside
            files = [f for f in os.listdir(path_csv) if f.endswith(".csv")]
            if files:
                print(f"Loading {filename} (from folder/file)...")
                return pd.read_csv(os.path.join(path_csv, files[0]), low_memory=False)

        # Case B: It is a standard file
        else:
            print(f"Loading {filename} (csv)...")
            return pd.read_csv(path_csv, low_memory=False)

    print(f"Warning: Could not find {filename}")
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 1. BUILD THE STATIC PROFILES (The "Skeleton")
    # ---------------------------------------------------------
    print("Building Patient Profiles...")

    # Load foundational tables
    patients = load_csv('hosp', 'patients')
    admissions = load_csv('hosp', 'admissions')
    icustays = load_csv('icu', 'icustays')

    if patients is None or admissions is None:
        print("Error: Missing base tables (patients/admissions). Exiting.")
        return

    # Filter for valid admissions in our cohort
    cohort_hadms = admissions['hadm_id'].unique()
    print(f"Creating folders for {len(cohort_hadms)} admissions...")

    # Pre-calculate stays for faster lookup
    if icustays is not None:
        icu_lookup = icustays.groupby('hadm_id')

    for hadm_id in tqdm(cohort_hadms):
        # Get Admission Info
        adm_row = admissions[admissions['hadm_id'] == hadm_id].iloc[0]
        subj_id = adm_row['subject_id']

        # Get Patient Info
        pat_rows = patients[patients['subject_id'] == subj_id]
        if pat_rows.empty: continue
        pat_row = pat_rows.iloc[0]

        # Create Folder: Processed_Patients / {subject_id} / {hadm_id}
        save_dir = os.path.join(OUTPUT_DIR, str(subj_id), str(hadm_id))
        os.makedirs(save_dir, exist_ok=True)

        # Construct static_profile.json
        profile = {
            "subject_id": int(subj_id),
            "hadm_id": int(hadm_id),
            "gender": pat_row['gender'],
            "anchor_age": int(pat_row['anchor_age']),
            "admittime": adm_row['admittime'],
            "dischtime": adm_row['dischtime'],
            "diagnosis": str(adm_row['diagnosis']) if 'diagnosis' in adm_row else "UNKNOWN",
            "icu_stays": []
        }

        # Add ICU Stay mappings
        if icustays is not None and hadm_id in icu_lookup.groups:
            stays = icu_lookup.get_group(hadm_id)
            for _, stay in stays.iterrows():
                profile["icu_stays"].append({
                    "stay_id": int(stay['stay_id']),
                    "intime": stay['intime'],
                    "outtime": stay['outtime'],
                    "los": stay['los']
                })

        # Save JSON
        with open(os.path.join(save_dir, 'static_profile.json'), 'w') as f:
            json.dump(profile, f, indent=4)

    # Cleanup memory
    del patients, admissions, icustays
    gc.collect()

    # ---------------------------------------------------------
    # 2. DISPATCH DATA TABLES (The "Meat")
    # ---------------------------------------------------------

    def distribute_table(folder, filename, output_name, cols_to_keep=None):
        """Reads a big table and saves slices to patient folders."""
        df = load_csv(folder, filename)
        if df is None: return

        print(f"Distributing {filename} to patient folders...")

        # Ensure we have hadm_id to split by
        if 'hadm_id' not in df.columns:
            print(f"Skipping {filename}: No hadm_id column.")
            return

        df = df.dropna(subset=['hadm_id'])

        # Optimize: Convert hadm_id to int to match folder structure
        df['hadm_id'] = df['hadm_id'].astype(int)

        # Group by hadm_id
        grouped = df.groupby('hadm_id')

        for hadm_id, group in tqdm(grouped):
            # We need subject_id to find the folder.
            if 'subject_id' in group.columns:
                subj_id = int(group.iloc[0]['subject_id'])
            else:
                continue

            target_dir = os.path.join(OUTPUT_DIR, str(subj_id), str(hadm_id))

            # Only save if the folder exists (i.e., it's a Sepsis admission)
            if os.path.exists(target_dir):
                save_path = os.path.join(target_dir, output_name)

                # Select columns if specified
                if cols_to_keep:
                    # Only keep columns that actually exist in this dataframe
                    valid_cols = [c for c in cols_to_keep if c in group.columns]
                    group_slice = group[valid_cols]
                else:
                    group_slice = group

                group_slice.to_csv(save_path, index=False)

        del df
        gc.collect()

    # --- A. VITALS (Rule 7) ---
    distribute_table('icu', 'chartevents', 'vitals.csv',
                     cols_to_keep=['charttime', 'stay_id', 'itemid', 'valuenum', 'valueuom'])

    # --- B. LABS (Rules 3, 4, 8) ---
    distribute_table('hosp', 'labevents', 'labs.csv',
                     cols_to_keep=['charttime', 'itemid', 'valuenum', 'flag'])

    # --- C. INPUTS (Rules 5, 6) ---
    distribute_table('icu', 'inputevents', 'inputs.csv',
                     cols_to_keep=['starttime', 'endtime', 'stay_id', 'itemid', 'amount',
                                   'amountuom', 'rate', 'rateuom', 'patientweight', 'ordercategoryname'])

    # --- D. MEDS ORDERED (Rule 2 Context) ---
    distribute_table('hosp', 'prescriptions', 'medications_ordered.csv',
                     cols_to_keep=['starttime', 'stoptime', 'drug', 'dose_val_rx', 'dose_unit_rx', 'route'])

    # --- E. MICROBIOLOGY (Rule 1) ---
    distribute_table('hosp', 'microbiologyevents', 'microbiology.csv',
                     cols_to_keep=['charttime', 'spec_type_desc', 'org_name', 'interpretation'])

    # --- F. MEDS ADMINISTERED (Actual Rule 2 Timestamp) ---
    distribute_table('hosp', 'emar', 'medications_admin.csv',
                     cols_to_keep=['charttime', 'medication', 'event_txt', 'scheduletime'])

    print("Done! Data sharding complete.")


if __name__ == "__main__":
    main()