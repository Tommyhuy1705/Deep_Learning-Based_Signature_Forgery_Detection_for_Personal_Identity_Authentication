import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, auc, confusion_matrix
)
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import os

# Ensure necessary imports from project structure
# Adjust relative paths if needed
try:
    from dataloader.meta_dataloader import SignatureEpisodeDataset
except ImportError:
    print("Warning: Could not import SignatureEpisodeDataset. Ensure dataloader path is correct.")

def calculate_far_frr_eer(true_labels, distances):
    """
    Calculates False Acceptance Rate (FAR), False Rejection Rate (FRR)
    across a range of thresholds, and determines the Equal Error Rate (EER).

    Args:
        true_labels (list or np.array): Aggregated true binary labels (0=Forgery, 1=Genuine).
        distances (list or np.array): Aggregated raw distances. Lower distance
                                      indicates higher likelihood of being genuine (class 1).

    Returns:
        tuple: A tuple containing:
            - float: Equal Error Rate (EER).
            - float: Threshold at which EER occurs.
            - np.array: Array of threshold values tested.
            - np.array: Array of FAR values corresponding to thresholds.
            - np.array: Array of FRR values corresponding to thresholds.
            Returns (None, None, None, None, None) if calculation is not possible.
    """
    true_labels = np.array(true_labels)
    distances = np.array(distances)

    # Ensure finite values
    finite_mask = np.isfinite(distances)
    if not np.any(finite_mask):
        print("Warning: No finite distances found for EER calculation.")
        return None, None, None, None, None

    true_labels = true_labels[finite_mask]
    distances = distances[finite_mask]

    if len(np.unique(true_labels)) < 2:
        print("Warning: EER requires both genuine and forged samples.")
        return None, None, None, None, None
    if len(distances) == 0:
         print("Warning: No valid distances for EER calculation.")
         return None, None, None, None, None


    # Generate thresholds based on sorted unique distances
    thresholds = np.sort(np.unique(distances))
    # Add points slightly below min and above max to cover all ranges
    thresholds = np.concatenate(([thresholds[0] - 1e-6], thresholds, [thresholds[-1] + 1e-6]))
    # More robust: Generate N thresholds linearly spaced between min and max
    # min_dist, max_dist = np.min(distances), np.max(distances)
    # thresholds = np.linspace(min_dist - 1e-6, max_dist + 1e-6, num=500) # Example: 500 thresholds

    far_list = []
    frr_list = []

    for thresh in thresholds:
        # Prediction: 1 (Genuine) if distance < threshold, 0 (Forgery) otherwise
        predictions = (distances < thresh).astype(int)

        # Calculate TP, FP, TN, FN
        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        tn = np.sum((predictions == 0) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))

        # Calculate FAR and FRR, handle division by zero
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        far_list.append(far)
        frr_list.append(frr)

    far_list = np.array(far_list)
    frr_list = np.array(frr_list)

    # Find the Equal Error Rate (EER)
    # EER occurs where FAR is approximately equal to FRR
    # Find the index where the absolute difference |FAR - FRR| is minimal
    eer_index = np.nanargmin(np.abs(far_list - frr_list))
    # EER is the value of FAR (or FRR) at this index (or average)
    eer = (far_list[eer_index] + frr_list[eer_index]) / 2.0
    eer_threshold = thresholds[eer_index]

    return eer, eer_threshold, thresholds, far_list, frr_list

def evaluate_meta_model(feature_extractor, metric_generator, test_dataset, device):
    """
    Evaluates the meta-learning model, now returning comprehensive metrics including EER.

    Args:
        feature_extractor (torch.nn.Module): The feature extractor model.
        metric_generator (torch.nn.Module): The metric generator model.
        test_dataset (Dataset): A SignatureEpisodeDataset instance for meta-testing.
        device (torch.device): The device (CPU or CUDA).

    Returns:
        tuple: A tuple containing:
            - dict: Dictionary with metrics: 'accuracy', 'precision', 'recall',
                    'f1_score', 'roc_auc', 'eer', 'eer_threshold'.
            - list: Aggregated true labels.
            - list: Aggregated predictions (based on per-episode optimal threshold).
            - list: Aggregated raw distances.
    """
    feature_extractor.eval()
    metric_generator.eval()

    num_workers = os.cpu_count() // 2 if 'kaggle' in os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') else 0
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    all_true_labels = []
    all_predictions = []
    all_distances = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Meta-Testing", leave=False):
            # --- (Data loading, Embedding Extraction, Metric Generation - SAME AS BEFORE) ---
            support_images = batch['support_images'].squeeze(0).to(device)
            query_images = batch['query_images'].squeeze(0).to(device)
            query_labels = batch['query_labels'].squeeze(0).to(device)
            user_id = batch.get('user_id', ['N/A'])[0]

            k_shot = len(support_images)
            if k_shot == 0: continue
            all_images = torch.cat([support_images, query_images], dim=0)
            try: all_embeddings = feature_extractor(all_images)
            except RuntimeError: continue
            support_embeddings = all_embeddings[:k_shot]
            query_embeddings = all_embeddings[k_shot:]
            if len(query_embeddings) == 0: continue
            try:
                W = metric_generator(support_embeddings)
                prototype_genuine = torch.mean(support_embeddings, dim=0)
            except RuntimeError: continue

            distances_episode = []
            valid_episode = True
            for q_embed in query_embeddings:
                diff = q_embed - prototype_genuine
                try:
                    dist = torch.matmul(torch.matmul(diff.unsqueeze(0), W), diff.unsqueeze(1)).item()
                    if not np.isfinite(dist): dist = torch.linalg.norm(diff).item()
                    distances_episode.append(dist)
                except Exception:
                    dist = torch.linalg.norm(diff).item()
                    distances_episode.append(dist)
                    valid_episode = False # Mark if errors occurred

            if not valid_episode: continue # Skip if distance calc failed badly

            distances_episode = np.array(distances_episode)
            labels_episode = query_labels.cpu().numpy()

            if distances_episode.size == 0 or labels_episode.size == 0 or distances_episode.size != labels_episode.size: continue

            # --- Find Optimal Threshold FOR THIS EPISODE ---
            best_acc_episode = -1.0
            default_thresh = np.median(distances_episode) if len(distances_episode) > 0 else 0.5
            best_thresh_episode = default_thresh
            sorted_dists = np.sort(np.unique(distances_episode))
            threshold_candidates = (sorted_dists[:-1] + sorted_dists[1:]) / 2.0
            if len(sorted_dists) > 0:
                 min_dist, max_dist = sorted_dists[0], sorted_dists[-1]
                 threshold_candidates = np.concatenate(([min_dist - 1e-6], threshold_candidates, [max_dist + 1e-6]))
            if len(threshold_candidates) == 0: threshold_candidates = [default_thresh]
            valid_thresholds = [th for th in threshold_candidates if np.isfinite(th)]
            if not valid_thresholds: valid_thresholds = [default_thresh]

            for thresh in valid_thresholds:
                preds = (distances_episode < thresh).astype(int)
                acc = np.mean(preds == labels_episode)
                if acc > best_acc_episode:
                    best_acc_episode = acc
                    best_thresh_episode = thresh
                elif acc == best_acc_episode:
                    current_median_diff = abs(best_thresh_episode - default_thresh)
                    new_median_diff = abs(thresh - default_thresh)
                    if new_median_diff < current_median_diff: best_thresh_episode = thresh

            # --- Generate Predictions & Aggregate ---
            final_preds_episode = (distances_episode < best_thresh_episode).astype(int)
            all_true_labels.extend(labels_episode.tolist())
            all_predictions.extend(final_preds_episode.tolist())
            all_distances.extend(distances_episode.tolist())

    # --- Calculate Overall Metrics ---
    if not all_true_labels or not all_predictions:
        print("Warning: No valid data aggregated for final metrics calculation.")
        zero_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'roc_auc': 0.0, 'eer': 1.0, 'eer_threshold': np.nan}
        return zero_metrics, [], [], []

    # Calculate standard metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_true_labels, all_predictions, zero_division=0)

    # Calculate ROC AUC
    roc_scores = -np.array(all_distances) # Higher score = more likely genuine
    valid_indices = np.isfinite(roc_scores)
    roc_auc = 0.0
    if np.any(valid_indices) and len(np.unique(np.array(all_true_labels)[valid_indices])) > 1:
        roc_auc = roc_auc_score(np.array(all_true_labels)[valid_indices], roc_scores[valid_indices])

    # Calculate EER
    eer, eer_threshold, _, _, _ = calculate_far_frr_eer(all_true_labels, all_distances)
    if eer is None: # Handle case where EER calculation failed
        eer = 1.0 # Worst case EER
        eer_threshold = np.nan

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'eer': eer,
        'eer_threshold': eer_threshold
    }

    return results, all_true_labels, all_predictions, all_distances

def plot_roc_curve(all_true_labels, all_distances, title='Receiver Operating Characteristic (ROC) Curve'):
    """ Plots the ROC curve and displays the AUC score. """
    # --- (Implementation unchanged, ensure it uses NEGATIVE distances for scores) ---
    if not all_true_labels or not all_distances or len(all_true_labels) != len(all_distances):
        print("Error: Invalid input data for ROC curve plotting.")
        return
    if len(np.unique(all_true_labels)) < 2:
        print("Warning: ROC AUC score is not defined when only one class is present.")
        # Optionally plot a dummy line or just return
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No discrimination')
        plt.title(title + " (Only one class present)")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
        return

    roc_scores = -np.array(all_distances)
    valid_indices = np.isfinite(roc_scores)
    if not np.all(valid_indices):
         # print(f"Warning: Removing {np.sum(~valid_indices)} non-finite scores before plotting ROC.") # Reduce verbosity
         roc_scores = roc_scores[valid_indices]
         all_true_labels = np.array(all_true_labels)[valid_indices]

    if len(np.unique(all_true_labels)) < 2: # Check again after filtering
        print("Warning: Still only one class present after filtering non-finite scores for ROC.")
        # Plot dummy line
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No discrimination')
        plt.title(title + " (Only one class after filtering)")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
        return

    fpr, tpr, _ = roc_curve(all_true_labels, roc_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1 - FRR)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_confusion_matrix(all_true_labels, all_predictions, class_names=['Forgery (0)', 'Genuine (1)'], title='Confusion Matrix'):
    """ Plots the confusion matrix using seaborn heatmap. """
    # --- (Implementation unchanged) ---
    if not all_true_labels or not all_predictions or len(all_true_labels) != len(all_predictions):
        print("Error: Invalid input data for confusion matrix plotting.")
        return

    cm = confusion_matrix(all_true_labels, all_predictions)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14}) # Increase annot size
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10, rotation=0)
    plt.show()


def plot_far_frr_eer(true_labels, distances, title='FAR/FRR vs. Threshold with EER'):
    """
    Calculates and plots FAR and FRR curves against distance thresholds,
    highlighting the Equal Error Rate (EER).

    Args:
        true_labels (list or np.array): Aggregated true binary labels (0=Forgery, 1=Genuine).
        distances (list or np.array): Aggregated raw distances.
        title (str): The title for the plot.
    """
    eer, eer_threshold, thresholds, far_list, frr_list = calculate_far_frr_eer(true_labels, distances)

    if eer is None or thresholds is None:
        print("Could not calculate or plot FAR/FRR/EER.")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, far_list, label='FAR (False Acceptance Rate)', color='red')
    plt.plot(thresholds, frr_list, label='FRR (False Rejection Rate)', color='blue')

    # Mark the EER point
    plt.plot(eer_threshold, eer, 'o', color='black', markersize=8, label=f'EER ≈ {eer:.4f} at threshold ≈ {eer_threshold:.4f}')

    # Find the intersection point visually more accurately if lines cross cleanly
    idx = np.argmin(np.abs(far_list - frr_list))
    plt.plot(thresholds[idx], far_list[idx], 'x', color='green', markersize=10) # Mark intersection point found by argmin

    plt.xlabel('Distance Threshold')
    plt.ylabel('Error Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # Adjust x-axis limits if needed based on threshold range
    # plt.xlim([min(thresholds)-0.1, max(thresholds)+0.1])
    plt.ylim([0.0, 1.05]) # Error rates are between 0 and 1
    plt.show()