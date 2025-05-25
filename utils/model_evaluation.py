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

# Hàm tính khoảng cách dựa trên metric được chọn
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
        raise ValueError(f"Metric không được hỗ trợ: {metric}")

# Hàm đánh giá mô hình
def evaluate_model(model: tSSN, metric, dataloader: DataLoader, device):
    model.eval()
    distances_list = []
    labels_list = []

    with torch.no_grad():
        for (anchor, positive, negative) in tqdm(dataloader, desc=f'Evaluating with {metric}'):
            # Chuyển dữ liệu sang device
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            # Trích xuất đặc trưng bằng phương thức forward
            anchor_feat, positive_feat, negative_feat = model(anchor, positive, negative)
            
            # Tính khoảng cách
            dist_ap = calculate_distance(anchor_feat, positive_feat, metric)
            dist_an = calculate_distance(anchor_feat, negative_feat, metric)
            
            # Thu thập khoảng cách và nhãn
            distances_list.extend(dist_ap.cpu().numpy().tolist())
            labels_list.extend([1] * dist_ap.size(0))  # anchor-positive: giống nhau
            distances_list.extend(dist_an.cpu().numpy().tolist())
            labels_list.extend([0] * dist_an.size(0))  # anchor-negative: khác nhau

    # Chuyển sang numpy array
    distances = np.array(distances_list)
    labels = np.array(labels_list)

    # Tìm ngưỡng tối ưu bằng ROC curve
    fpr, tpr, thresholds = roc_curve(labels, -distances)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = -thresholds[optimal_idx]

    # Tính các chỉ số hiệu suất
    predictions = (distances <= optimal_threshold).astype(int)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    roc_auc = auc(fpr, tpr)

    # Tính FAR và FRR tại ngưỡng tối ưu
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Tính EER bằng cách phân tích FAR và FRR trên dải ngưỡng
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

    # Tìm EER
    diff = np.abs(np.array(far_list) - np.array(frr_list))
    eer_index = np.argmin(diff)
    eer = (far_list[eer_index] + frr_list[eer_index]) / 2
    eer_threshold = threshold_range[eer_index]

    # Kết quả trả về
    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'threshold': optimal_threshold,
        'y_true': labels,
        'distances': distances,
        'far': far,                # FAR tại ngưỡng tối ưu
        'frr': frr,                # FRR tại ngưỡng tối ưu
        'eer': eer,                # Equal Error Rate
        'eer_threshold': eer_threshold,
        'threshold_range': threshold_range,
        'far_list': far_list,
        'frr_list': frr_list
    }
    return result

# Hàm vẽ đồ thị tìm best accurracy
def draw_plot_find_acc(results_dict):
    keys = list(results_dict.keys())
    accuracies = [results_dict[k]['mean_acc'] for k in keys]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(keys, accuracies, marker='o')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Mode_Margin')
    plt.ylabel('Mean Accuracy')
    plt.title('Mean Accuracy for Different Modes and Margins')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Tìm best
    best_key = keys[accuracies.index(max(accuracies))]
    best_acc = max(accuracies)

    print(f"\nBest model: {best_key} | Mean Accuracy: {best_acc:.4f}")

    # Tách key để lấy mode và margin
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
        raise ValueError("results must be a dictionary or a list of dictionaries")
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
    # Accuracy, Precision, Recall, F1
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [results_df['accuracy'].iloc[0], results_df['precision'].iloc[0],
            results_df['recall'].iloc[0], results_df['f1'].iloc[0]]
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.title('Các Chỉ Số Đánh Giá Mô Hình')
    plt.show()
    

def draw_confusion_matrix(results_df, results):
    # Confusion Matrix
    threshold = results_df['threshold'].iloc[0]
    y_pred = [1 if d < threshold else 0 for d in results['distances']]
    cm = confusion_matrix(results['y_true'], y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Dự Đoán')
    plt.ylabel('Thực Tế')
    plt.title('Ma Trận Nhầm Lẫn')
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
    plt.title('Đường Cong ROC')
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
    plt.title('Đường Cong Precision-Recall')
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
