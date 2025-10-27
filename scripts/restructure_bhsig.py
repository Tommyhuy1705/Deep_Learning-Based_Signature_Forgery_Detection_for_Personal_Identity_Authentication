import os
import re
import json
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

def parse_bhsig_filename(filename: str) -> Optional[Tuple[str, int]]:
    """
    Extracts language code ('B' or 'H') and user ID from BHSig-style filenames.
    Example: 'B-S-001-G-01.tif' -> ('B', 1)  (Handles potential leading zeros)
             'H-S-042-F-03.tif' -> ('H', 42)

    Args:
        filename (str): The input filename.

    Returns:
        Optional[Tuple[str, int]]: A tuple (language_code, user_id), or None if parsing fails.
    """
    # Regex updated to capture Language (B or H) and User ID (digits) from filename
    # Allows for variations like -S-, -G-, -F- and leading zeros in user ID
    match = re.match(r'^([BH])-[SFG]-(\d+)-[GF]-(\d+)\.tif$', filename, re.IGNORECASE)
    if match:
        language_code = match.group(1).upper() # 'B' or 'H'
        user_id = int(match.group(2)) # Convert captured digits to integer
        return language_code, user_id
    else:
        # print(f"Warning: Could not parse BHSig filename format: {filename}")
        return None

def restructure_nth2165_bhsig(base_dir: str, output_dir: str, num_bengali_test: int = 50, num_hindi_test: int = 30, seed: int = 42):
    """
    Scans the 'nth2165/bhsig260-hindi-bengali' dataset based on its specific structure,
    separates Bengali and Hindi users, and creates two separate JSON files
    ('bhsig_bengali_meta_test.json', 'bhsig_hindi_meta_test.json')
    containing absolute paths for randomly selected test users.

    Args:
        base_dir (str): Path to the root directory of the 'nth2165/bhsig260-hindi-bengali' dataset.
                        (e.g., '/kaggle/input/bhsig260-hindi-bengali/')
        output_dir (str): Directory where the restructured JSON split files will be saved.
        num_bengali_test (int): Number of Bengali users to include in the Bengali test split. Defaults to 50.
        num_hindi_test (int): Number of Hindi users to include in the Hindi test split. Defaults to 30.
        seed (int): Random seed for selecting test users. Defaults to 42.
    """
    print("--- Starting BHSig-260 (nth2165) Dataset Restructuring ---")

    # --- Define Source Directories based on the CORRECTED structure ---
    source_dirs = {
        'hindi_genuine': os.path.join(base_dir, 'BHSig160_Hindi', 'Genuine'),
        'hindi_forged': os.path.join(base_dir, 'BHSig160_Hindi', 'Forged'),
        'bengali_genuine': os.path.join(base_dir, 'BHSig100_Bengali', 'Genuine'),
        'bengali_forged': os.path.join(base_dir, 'BHSig100_Bengali', 'Forged')
    }

    # --- 1. Validate Input Directories ---
    all_dirs_exist = True
    for key, directory in source_dirs.items():
        if not os.path.isdir(directory):
            print(f"ERROR: Directory not found: {directory}. Please check the 'base_dir' path.")
            all_dirs_exist = False
    if not all_dirs_exist:
        return # Stop if essential directories are missing

    # --- 2. Collect All Files Grouped by Unique User ID ('B-1', 'H-42', etc.) ---
    all_files: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {'genuine': [], 'forgery': []})
    processed_files_count = 0
    skipped_files_count = 0
    supported_extensions = ('.tif', '.tiff') # Dataset uses .tif

    print("Scanning source directories...")
    for key, directory in source_dirs.items():
        print(f"  Processing: {directory}")
        for filename in os.listdir(directory):
            if filename.lower().endswith(supported_extensions):
                parsed_info = parse_bhsig_filename(filename)
                if parsed_info:
                    lang, user_id = parsed_info
                    # Create unique ID (e.g., 'B-1', 'H-11') consistent across genuine/forged
                    unique_user_id = f"{lang}-{user_id}"
                    full_path = os.path.join(directory, filename)

                    # Add file path to the correct category (genuine/forgery)
                    if 'genuine' in key:
                        all_files[unique_user_id]['genuine'].append(full_path)
                    elif 'forged' in key:
                        all_files[unique_user_id]['forgery'].append(full_path)
                    processed_files_count += 1
                else:
                    skipped_files_count += 1 # Skip files with unparsable names
            else:
                 skipped_files_count += 1 # Skip non-image files

    if not all_files:
         print("ERROR: No valid BHSig files found or parsed. Cannot create splits.")
         return

    print(f"Finished scanning. Processed {processed_files_count} files, skipped {skipped_files_count}.")
    print(f"Found data for {len(all_files)} unique users (Bengali + Hindi).")

    # --- 3. Separate Users by Language and Select Test Subsets ---
    bengali_users = sorted([uid for uid in all_files if uid.startswith('B-')])
    hindi_users = sorted([uid for uid in all_files if uid.startswith('H-')])

    print(f"  Available Bengali users: {len(bengali_users)}")
    print(f"  Available Hindi users: {len(hindi_users)}")

    # Adjust requested numbers if fewer users are available
    num_bengali_test = min(num_bengali_test, len(bengali_users))
    num_hindi_test = min(num_hindi_test, len(hindi_users))

    random.seed(seed)
    random.shuffle(bengali_users)
    random.shuffle(hindi_users)

    test_bengali_ids = bengali_users[:num_bengali_test]
    test_hindi_ids = hindi_users[:num_hindi_test]

    print(f"Selected {len(test_bengali_ids)} Bengali users for Bengali test split.")
    print(f"Selected {len(test_hindi_ids)} Hindi users for Hindi test split.")

    # --- 4. Create and Save Separate JSON Files ---
    os.makedirs(output_dir, exist_ok=True)

    # Save Bengali Split
    if test_bengali_ids:
        bengali_meta_test_data = {uid: all_files[uid] for uid in test_bengali_ids}
        bengali_split_data = {'meta-test': bengali_meta_test_data}
        bengali_output_path = os.path.join(output_dir, 'bhsig_bengali_meta_test.json')
        try:
            with open(bengali_output_path, 'w', encoding='utf-8') as f:
                json.dump(bengali_split_data, f, indent=4)
            print(f"  Successfully saved Bengali meta-test split to: {bengali_output_path}")
        except Exception as e:
            print(f"  ERROR saving Bengali split file: {e}")
    else:
        print("  No Bengali users selected, skipping Bengali split file generation.")

    # Save Hindi Split
    if test_hindi_ids:
        hindi_meta_test_data = {uid: all_files[uid] for uid in test_hindi_ids}
        hindi_split_data = {'meta-test': hindi_meta_test_data}
        hindi_output_path = os.path.join(output_dir, 'bhsig_hindi_meta_test.json')
        try:
            with open(hindi_output_path, 'w', encoding='utf-8') as f:
                json.dump(hindi_split_data, f, indent=4)
            print(f"  Successfully saved Hindi meta-test split to: {hindi_output_path}")
        except Exception as e:
            print(f"  ERROR saving Hindi split file: {e}")
    else:
        print("  No Hindi users selected, skipping Hindi split file generation.")

    print("--- BHSig-260 Restructuring Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Restructure the 'nth2165/bhsig260-hindi-bengali' dataset "
                                                 "into separate Bengali and Hindi meta-test JSON files.")
    parser.add_argument('--base_dir', type=str, required=True,
                        help="Path to the root directory of the 'nth2165/bhsig260-hindi-bengali' dataset "
                             "(e.g., /kaggle/input/bhsig260-hindi-bengali).")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the output JSON split files (e.g., /kaggle/working/).")
    parser.add_argument('--num_bengali', type=int, default=50,
                        help="Number of Bengali users for the Bengali 'meta-test' split.")
    parser.add_argument('--num_hindi', type=int, default=30,
                        help="Number of Hindi users for the Hindi 'meta-test' split.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for selecting test users.")

    args = parser.parse_args()
    restructure_nth2165_bhsig(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        num_bengali_test=args.num_bengali,
        num_hindi_test=args.num_hindi,
        seed=args.seed
    )