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

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'threshold': optimal_threshold,
        'y_true': labels,
        'distances': distances
    }

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
    print(results_df)

    # Create the plot
    if req == 'acc':
        draw_acc(results_df)
    elif req == "f1":
        draw_f1(results_df)
    elif req == "roc-auc":
        draw_roc_auc(results_df)
    elif req == "all":
        if req == "all":
            # Accuracy, Precision, Recall, F1
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [results_df['accuracy'].iloc[0], results_df['precision'].iloc[0],
                    results_df['recall'].iloc[0], results_df['f1'].iloc[0]]
            plt.figure(figsize=(8, 6))
            plt.bar(metrics, values)
            plt.ylim(0, 1)
            plt.title('Các Chỉ Số Đánh Giá Mô Hình')
            plt.grid(True)  # Thêm lưới
            plt.show()
            
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
            plt.grid(True)  # Thêm lưới
            plt.show()
            
            # Confusion Matrix
            threshold = results_df['threshold'].iloc[0]
            y_pred = [1 if d < threshold else 0 for d in results['distances']]
            cm = confusion_matrix(results['y_true'], y_pred)
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Dự Đoán')
            plt.ylabel('Thực Tế')
            plt.title('Ma Trận Nhầm Lẫn')
            plt.grid(True)  # Thêm lưới
            plt.show()
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(results['y_true'], -results['distances'])
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Đường Cong Precision-Recall')
            plt.grid(True)  # Thêm lưới
            plt.show()



    def draw_acc(results_df):
        """"Vẽ biểu đồ so sánh Accuracy của từng Mode"""
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=results_df, x='margin', y='accuracy', hue='metric', marker='o')
        plt.title('Accuracy vs Margin for each Metric')
        plt.xlabel('Margin')
        plt.ylabel('Accuracy')
        plt.grid(True)  # Thêm lưới
        plt.show()
        

    def draw_f1(results_df):
        """Vẽ biểu đồ so sánh F1-score"""
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=results_df.index, y='f1', data=results_df)

        # Thêm nhãn giá trị lên từng cột
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=5, color='black', fontweight='bold')

        plt.xticks(rotation=45)
        plt.title('F1-Score Comparison Across Metrics and Margins')
        plt.xlabel('Metric_Margin')
        plt.ylabel('F1-Score')
        plt.tight_layout()
        plt.show()

    def draw_roc_auc(results_df):
        """Vẽ biểu đồ so sánh ROC-AUC"""
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=results_df.index, y='roc_auc', data=results_df)

        # Thêm nhãn giá trị lên từng cột
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=5, color='black', fontweight='bold')

        plt.xticks(rotation=45)
        plt.title('ROC-AUC Comparison Across Metrics and Margins')
        plt.xlabel('Metric_Margin')
        plt.ylabel('ROC-AUC')
        plt.tight_layout()
        plt.grid(True)  # Thêm lưới
        plt.show()
