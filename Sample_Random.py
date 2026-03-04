r"""
=============================================================================
Patient Cohort Sampler
=============================================================================
Randomly samples a subset of patients from the processed MIMIC-IV cohort
for downstream LLM and Fuzzy inference processing. Ensures reproducibility
using a fixed random seed.

Usage:
  python sample_patients.py --sample_size 2000 --seed 55
=============================================================================
"""

import pandas as pd
import os
import shutil
import random
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = r"."
IDS_FILE = os.path.join(BASE_DIR, "Septic_Mimic", "sepsis_ids.csv")
SOURCE_FOLDERS = os.path.join(BASE_DIR, "Processed_Patients")
TARGET_FOLDERS = os.path.join(BASE_DIR, "2000_Patients_Sampled")

SAMPLE_SIZE = 2000
RANDOM_SEED = 55  # Ensures reproducibility


def main():
    # 1. Load and Sample IDs
    # FIX: Check if file exists; if not, scan the folder structure directly.
    if os.path.exists(IDS_FILE):
        print(f"Loading Subject IDs from: {IDS_FILE}")
        df = pd.read_csv(IDS_FILE)
        all_subjects = df['subject_id'].unique().tolist()
    else:
        print(f"Warning: ID file not found at {IDS_FILE}")
        print(f"Falling back to scanning directory: {SOURCE_FOLDERS}")

        if not os.path.exists(SOURCE_FOLDERS):
            print(f"Error: Source folder '{SOURCE_FOLDERS}' does not exist. Cannot proceed.")
            return

        # Get all numbered folders (Subject IDs) from the source directory
        all_subjects = [int(d) for d in os.listdir(SOURCE_FOLDERS)
                        if os.path.isdir(os.path.join(SOURCE_FOLDERS, d)) and d.isdigit()]

    print(f"Total available subjects: {len(all_subjects)}")

    if len(all_subjects) < SAMPLE_SIZE:
        print(f"Warning: Requested {SAMPLE_SIZE} but only have {len(all_subjects)}. Using all.")
        sampled_subjects = all_subjects
    else:
        random.seed(RANDOM_SEED)
        sampled_subjects = random.sample(all_subjects, SAMPLE_SIZE)
        print(f"Randomly selected {len(sampled_subjects)} subjects (Seed: {RANDOM_SEED}).")

    # 2. Prepare Target Directory
    if os.path.exists(TARGET_FOLDERS):
        user_input = input(f"Target folder '{TARGET_FOLDERS}' exists. Delete and restart? (y/n): ")
        if user_input.lower() == 'y':
            shutil.rmtree(TARGET_FOLDERS)
        else:
            print("Aborted.")
            return

    os.makedirs(TARGET_FOLDERS, exist_ok=True)

    # 3. Copy Folders
    print("Copying patient data to sampled directory...")

    copied_patients = 0
    total_episodes = 0

    for subj_id in tqdm(sampled_subjects):
        src_path = os.path.join(SOURCE_FOLDERS, str(subj_id))
        dst_path = os.path.join(TARGET_FOLDERS, str(subj_id))

        # Check if the processed folder actually exists (it should)
        if os.path.exists(src_path):
            # Copy the entire patient folder (including all hadm_id subfolders)
            shutil.copytree(src_path, dst_path)
            copied_patients += 1

            # Count episodes (subdirectories)
            episodes = [d for d in os.listdir(dst_path) if os.path.isdir(os.path.join(dst_path, d))]
            total_episodes += len(episodes)

    # 4. Summary
    print("\n" + "=" * 40)
    print("SAMPLING COMPLETE")
    print("=" * 40)
    print(f"Source: {SOURCE_FOLDERS}")
    print(f"Target: {TARGET_FOLDERS}")
    print(f"Patients Sampled: {len(sampled_subjects)}")
    print(f"Patients Copied:  {copied_patients} (Found in source)")
    print(f"Total Episodes:   {total_episodes}")
    print("=" * 40)
    print("You can now run your LLM/Fuzzy scripts on the 'Target' folder.")


if __name__ == "__main__":
    main()