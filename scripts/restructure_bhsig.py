import os
import re
import json
import random
import argparse
from collections import defaultdict
from sklearn.model_selection import KFold

def parse_bhsig_filename(filename):
    """
    Parses BHSig filename to extract User ID.
    Format example: 'B-S-001-G-01.tif' -> 'B-1'
    """
    match = re.match(r'^([BH])-[SFG]-(\d+)-[GF]-(\d+)\.tif$', filename, re.IGNORECASE)
    if match:
        return f"{match.group(1)}-{int(match.group(2))}" # e.g., 'B-1' or 'H-42'
    return None

def restructure_bhsig_stratified(base_dir, output_dir, pretrain_users_count=150, n_folds=5, seed=42):
    """
    Splits BHSig dataset into:
    1. Background Set (for Pre-training): First 'pretrain_users_count' users.
    2. Evaluation Set (for Meta-learning): Remaining users, split into K-Folds.
    """
    print("--- Structuring BHSig: Splitting Background (Pretrain) vs Evaluation (Meta-train) ---")
    
    # Paths (Assuming standard structure inside BHSig260)
    # Check if paths exist, handle case sensitivity or slight naming variations if needed
    sub_dirs = [
        'BHSig160_Hindi/Genuine', 'BHSig160_Hindi/Forged',
        'BHSig100_Bengali/Genuine', 'BHSig100_Bengali/Forged'
    ]
    
    user_files = defaultdict(lambda: {'genuine': [], 'forgery': []})
    
    for sub in sub_dirs:
        full_path = os.path.join(base_dir, sub)
        if not os.path.isdir(full_path):
            print(f"Warning: Directory not found: {full_path}")
            continue
            
        print(f"Scanning: {full_path}")
        for f in os.listdir(full_path):
            if f.lower().endswith(('.tif', '.tiff', '.png')):
                uid = parse_bhsig_filename(f)
                if uid:
                    fpath = os.path.join(full_path, f)
                    # Determine if Genuine or Forged based on path name
                    if 'Genuine' in sub: 
                        user_files[uid]['genuine'].append(fpath)
                    else: 
                        user_files[uid]['forgery'].append(fpath)

    all_users = sorted(list(user_files.keys()))
    
    if len(all_users) == 0:
        print("ERROR: No users found. Check 'base_dir' path.")
        return

    # Shuffle users to mix Hindi and Bengali randomly
    random.seed(seed)
    random.shuffle(all_users)
    
    # --- SPLIT STRATEGY ---
    background_users = all_users[:pretrain_users_count]
    eval_users = all_users[pretrain_users_count:]
    
    print(f"Total Users Found: {len(all_users)}")
    print(f"Background Users (Pre-train): {len(background_users)}")
    print(f"Evaluation Users (Meta-train): {len(eval_users)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save Background Users List (for Pre-training Dataloader)
    bg_output = os.path.join(output_dir, 'bhsig_background_users.json')
    with open(bg_output, 'w') as f:
        json.dump(background_users, f, indent=4)
    print(f"Saved Pre-train user list to: {bg_output}")
        
    # 2. Create K-Fold Splits for Evaluation Users (Meta-Learning)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(eval_users)):
        train_uids = [eval_users[i] for i in train_idx]
        val_uids = [eval_users[i] for i in val_idx]
        
        split_data = {
            'meta-train': {uid: user_files[uid] for uid in train_uids},
            'meta-test':  {uid: user_files[uid] for uid in val_uids}
        }
        
        out_path = os.path.join(output_dir, f'bhsig_meta_split_fold_{fold}.json')
        with open(out_path, 'w') as f:
            json.dump(split_data, f, indent=4)
        print(f"  Fold {fold}: {len(train_uids)} Train / {len(val_uids)} Val users -> {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True, help="Root path of BHSig dataset")
    parser.add_argument('--output_dir', type=str, required=True, help="Output folder for JSON splits")
    parser.add_argument('--pretrain_users', type=int, default=150, help="Number of users for pre-training")
    args = parser.parse_args()
    
    restructure_bhsig_stratified(args.base_dir, args.output_dir, args.pretrain_users)