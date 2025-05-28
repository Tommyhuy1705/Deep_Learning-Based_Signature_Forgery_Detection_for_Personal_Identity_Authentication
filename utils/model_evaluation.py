import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.Triplet_Siamese_Similarity_Network import tSSN
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_curve, auc, precision_recall_curve, confusion_matrix
)
from losses.triplet_loss import DistanceNet

# Function to calculate distance based on selected metric
def calculate_distance(anchor_feat, test_feat, metric='euclidean', device='cuda'):
    distance_net = DistanceNet(input_dim=512).to(device)
    if metric == 'euclidean':
        return F.pairwise_distance(anchor_feat, test_feat)
    elif metric == 'cosine':
        anchor_feat = F.normalize(anchor_feat, p=2, dim=1)
        test_feat = F.normalize(test_feat, p=2, dim=1)
        return 1 - torch.sum(anchor_feat * test_feat, dim=1)
    elif metric == 'manhattan':
        return torch.sum(torch.abs(anchor_feat - test_feat), dim=1)
    elif metric == 'learnable':
        return distance_net(anchor_feat, test_feat)
    else:
        raise ValueError(f"Metrics not supported: {metric}")

# Model evaluation function
def evaluate_model(model: tSSN, metric, dataloader: DataLoader, device):
    model.eval()
    distances_list = []
    labels_list = []

    with torch.no_grad():
        for (anchor, positive, negative) in tqdm(dataloader, desc=f'Evaluating with {metric}'):
            # Convert data to device and extract features
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_feat, positive_feat, negative_feat = model(anchor, positive, negative)
            
            # Calculate distance
            dist_ap = calculate_distance(anchor_feat, positive_feat, metric)
            dist_an = calculate_distance(anchor_feat, negative_feat, metric)
            
            # Collect distances and labels
            # anchor-positive
            distances_list.extend(dist_ap.cpu().numpy().tolist())
            labels_list.extend([1] * dist_ap.size(0))
            # anchor-negative
            distances_list.extend(dist_an.cpu().numpy().tolist())
            labels_list.extend([0] * dist_an.size(0))

    # Convert to numpy array
    distances = np.array(distances_list)
    labels = np.array(labels_list)

    # Find the optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(labels, -distances)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = -thresholds[optimal_idx]

    # Calculate performance indicators
    predictions = (distances <= optimal_threshold).astype(int)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    roc_auc = auc(fpr, tpr)

    # Calculate FAR and FRR at optimal threshold
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Calculate EER (Equal Error Rate) by analyzing FAR and FRR over the threshold range
    min_dist, max_dist = np.min(distances), np.max(distances)
    threshold_range = np.linspace(min_dist, max_dist, 100)
    far_list, frr_list = [], []

    for thresh in threshold_range:
        preds = (distances <= thresh).astype(int)
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        far_val = fp / (fp + tn) if (fp + tn) > 0 else 0
        frr_val = fn / (fn + tp) if (fn + tp) > 0 else 0
        far_list.append(far_val)
        frr_list.append(frr_val)

    # Find EER (Equal Error Rate)
    diff = np.abs(np.array(far_list) - np.array(frr_list))
    eer_index = np.argmin(diff)
    eer = (far_list[eer_index] + frr_list[eer_index]) / 2
    eer_threshold = threshold_range[eer_index]

    # Return results
    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'threshold': optimal_threshold,
        'y_true': labels,
        'distances': distances,
        'far': far,
        'frr': frr,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'threshold_range': threshold_range,
        'far_list': far_list,
        'frr_list': frr_list
    }
    return result

# Graph function to find best accuracy
def draw_plot_find_acc(results_dict):
    keys = list(results_dict.keys())
    accuracies = [results_dict[k]['mean_acc'] for k in keys]

    plt.figure(figsize=(12, 6))
    plt.plot(keys, accuracies, marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Mode_Margin')
    plt.ylabel('Mean Accuracy')
    plt.title('Mean Accuracy for Different Modes and Margins')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    best_key = keys[accuracies.index(max(accuracies))]
    best_acc = max(accuracies)

    print(f"\nBest model: {best_key} | Mean Accuracy: {best_acc:.4f}")

    # Split key to get mode and margin
    if best_key == 'learnable':
        best_params = {'mode': 'learnable', 'margin': 0}
    else:
        mode, margin = best_key.split('_')
        best_params = {'mode': mode, 'margin': margin}

    return best_params

def draw_plot_evaluate(results, req=None):
    pd.set_option('display.width', 1000)
    pd.set_option('display.width', 1000)
    if isinstance(results, dict):
        results_df = pd.DataFrame([results])  # Single result
    elif isinstance(results, list):
        results_df = pd.DataFrame(results)   # Multiple results
    else:
        raise ValueError("Results must be a dictionary or a list of dictionaries")
    print('\nResults Table:')
    print(results_df.drop(columns=['y_true', 'distances', 'threshold_range', 'far_list', 'frr_list']))  # Loại bỏ cột dài

    # Create the plot
    if req == 'acc':
        draw_acc(results_df)
    elif req == "cm":
        draw_confusion_matrix(results_df, results)
    elif req == "roc-auc":
        draw_roc_auc(results_df)
    elif req == "pre-recall":
        draw_pre_recall(results)
    elif req == "far-frr":
        draw_far_frr(results)
    elif req == "all":
        draw_acc(results_df)
        draw_confusion_matrix(results_df, results)
        draw_roc_auc(results)
        draw_pre_recall(results)
        draw_far_frr(results)



def draw_acc(results_df):
    # List of metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    # Get values ​​from DataFrame
    values = [results_df['accuracy'].iloc[0], results_df['precision'].iloc[0],
              results_df['recall'].iloc[0], results_df['f1'].iloc[0]]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics, values, color='b')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)
    
    min_val = min(values)
    max_val = max(values)
    padding = (max_val - min_val) * 0.2
    if padding == 0:
        padding = 0.01
    plt.ylim(min_val - padding, max_val + padding)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title('Model Evaluation Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.show()
    

def draw_confusion_matrix(results_df, results):
    # Confusion Matrix
    threshold = results_df['threshold'].iloc[0]
    y_pred = [1 if d < threshold else 0 for d in results['distances']]
    cm = confusion_matrix(results['y_true'], y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prediction')
    plt.ylabel('Reality')
    plt.title('Confusion Matrix')
    plt.show()

def draw_roc_auc(results):
    # ROC Curve
    fpr, tpr, _ = roc_curve(results['y_true'], -results['distances'])
    roc_auc_value = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_value:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def draw_pre_recall(results):
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(results['y_true'], -results['distances'])
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.show()


def draw_far_frr(results):
    # FAR và FRR vs Threshold với EER
    plt.figure(figsize=(8, 6))
    plt.plot(results['threshold_range'], results['far_list'], label='FAR')
    plt.plot(results['threshold_range'], results['frr_list'], label='FRR')
    plt.axvline(x=results['eer_threshold'], color='r', linestyle='--', 
                label=f'EER Threshold: {results["eer_threshold"]:.2f} (EER = {results["eer"]:.4f})')
    plt.xlabel('Distance Threshold')
    plt.ylabel('Error Rate')
    plt.title('FAR và FRR vs Distance Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()
