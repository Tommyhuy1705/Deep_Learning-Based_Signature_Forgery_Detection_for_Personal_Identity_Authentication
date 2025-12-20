import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)
from tqdm.notebook import tqdm

# Set aesthetic style for all plots
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11})

def compute_metrics(y_true, y_distances):
    """
    Computes a comprehensive set of biometric verification metrics.
    """
    y_true = np.array(y_true)
    y_scores = -np.array(y_distances) # Convert distance to similarity score for ROC
    
    # 1. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    roc_auc = auc(fpr, tpr)
    
    # 2. EER Calculation (Intersection of FAR and FRR)
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_idx]
    optimal_threshold = -thresholds[eer_idx] # Revert sign back to distance
    
    # 3. Binary Classification Metrics at Optimal Threshold
    y_pred = (np.array(y_distances) < optimal_threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'eer': eer,
        'auc': roc_auc,
        'threshold': optimal_threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'fpr': fpr,
        'tpr': tpr,
        'fnr': fnr,
        'y_true': y_true,
        'y_pred': y_pred,
        'scores': y_distances
    }

def plot_comprehensive_evaluation(results, title_suffix="", save_path=None):
    """
    Generates a 2x2 dashboard of evaluation plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Comprehensive Evaluation Report {title_suffix}', fontsize=16, fontweight='bold')
    
    # --- 1. ROC Curve ---
    axes[0, 0].plot(results['fpr'], results['tpr'], color='darkorange', lw=2, label=f'AUC = {results["auc"]:.4f}')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 0].scatter(results['eer'], 1-results['eer'], color='red', label=f'EER = {results["eer"]:.2%}', zorder=5)
    axes[0, 0].set_xlabel('False Positive Rate (FAR)')
    axes[0, 0].set_ylabel('True Positive Rate (TAR)')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # --- 2. DET Curve (Log Scale) ---
    clip_min = 1e-4
    clip_fpr = np.clip(results['fpr'], clip_min, 1.0)
    clip_fnr = np.clip(results['fnr'], clip_min, 1.0)
    
    axes[0, 1].plot(clip_fpr, clip_fnr, color='green', lw=2)
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('False Acceptance Rate (FAR)')
    axes[0, 1].set_ylabel('False Rejection Rate (FRR)')
    axes[0, 1].set_title('Detection Error Trade-off (DET) Curve')
    axes[0, 1].grid(True, which="both", linestyle='--', alpha=0.4)

    # --- 3. Distance Distribution ---
    df = pd.DataFrame({'Distance': results['scores'], 'Label': results['y_true']})
    df['Type'] = df['Label'].apply(lambda x: 'Genuine Pairs' if x == 1 else 'Forgery Pairs')
    
    sns.histplot(data=df, x='Distance', hue='Type', kde=True, ax=axes[1, 0], 
                 palette={'Genuine Pairs': 'green', 'Forgery Pairs': 'red'}, 
                 bins=40, alpha=0.5, stat="density", common_norm=False)
    axes[1, 0].axvline(results['threshold'], color='blue', linestyle='--', lw=2, label=f'Threshold: {results["threshold"]:.3f}')
    axes[1, 0].set_title('Feature Distance Distribution')
    axes[1, 0].legend()

    # --- 4. Confusion Matrix ---
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=['Pred: Forged', 'Pred: Genuine'], 
                yticklabels=['True: Forged', 'True: Genuine'])
    axes[1, 1].set_title(f'Confusion Matrix (Acc: {results["accuracy"]:.2%})')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" > [Saved] Figure saved to: {save_path}")
        
    plt.show()

def visualize_hard_examples(feature_extractor, metric_generator, dataloader, device, threshold, num_examples=3, save_dir=None):
    """
    Visualizes False Acceptances and False Rejections.
    """
    feature_extractor.eval()
    metric_generator.eval()
    
    fa_cases = [] # False Accept (Impostor accepted)
    fr_cases = [] # False Reject (Genuine rejected)
    
    print(f"\n{'='*20} QUALITATIVE ERROR ANALYSIS {'='*20}")
    
    with torch.no_grad():
        for batch in dataloader:
            if len(fa_cases) >= num_examples and len(fr_cases) >= num_examples: break
            
            supports = batch['support_images'].to(device)
            queries = batch['query_images'].to(device)
            labels = batch['query_labels'].to(device)
            
            bs = supports.size(0)
            for i in range(bs):
                # Inference
                raw_s = feature_extractor(supports[i])
                raw_q = feature_extractor(queries[i])
                s_proto = F.normalize(raw_s, p=2, dim=1).mean(dim=0)
                q_feat = F.normalize(raw_q, p=2, dim=1)
                
                weights = metric_generator(s_proto.unsqueeze(0)).squeeze(0)
                
                dist_sq = (weights * (q_feat - s_proto).pow(2)).sum(dim=1)
                dists = torch.sqrt(torch.clamp(dist_sq, min=1e-8))
                
                # Undo normalization for visualization
                def denorm(tensor):
                    t = tensor.cpu().clone()
                    t = t * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                    return (1.0 - t).clamp(0, 1)

                # Iterate through 8 queries in this episode
                current_labels = labels[i]
                for k in range(len(current_labels)):
                    l = current_labels[k].item()
                    d = dists[k].item()
                    
                    pred_is_gen = d < threshold
                    
                    if l == 0 and pred_is_gen: # False Accept
                        if len(fa_cases) < num_examples:
                            fa_cases.append((denorm(supports[i][0]), denorm(queries[i][k]), d))
                    
                    elif l == 1 and not pred_is_gen: # False Reject
                        if len(fr_cases) < num_examples:
                            fr_cases.append((denorm(supports[i][0]), denorm(queries[i][k]), d))

    def plot_row(cases, title, filename_suffix):
        if not cases: return
        fig, axes = plt.subplots(len(cases), 2, figsize=(8, 3*len(cases)))
        if len(cases) == 1: axes = [axes] 
        
        for idx, (supp, query, d) in enumerate(cases):
            ax_s = axes[idx][0] if len(cases) > 1 else axes[0]
            ax_q = axes[idx][1] if len(cases) > 1 else axes[1]
            
            ax_s.imshow(supp.permute(1, 2, 0).numpy())
            ax_s.set_title("Reference (Support)")
            ax_s.axis('off')
            
            ax_q.imshow(query.permute(1, 2, 0).numpy())
            ax_q.set_title(f"Query (Dist: {d:.3f})")
            ax_q.axis('off')
            
        plt.suptitle(title, y=1.02, fontsize=14, color='darkred', fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            path = os.path.join(save_dir, f"error_analysis_{filename_suffix}.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f" > [Saved] Error analysis saved to: {path}")
            
        plt.show()

    if fa_cases: plot_row(fa_cases, f"FALSE ACCEPTANCE (Forgery Accepted)\nThreshold: {threshold:.3f}", "false_accept")
    if fr_cases: plot_row(fr_cases, f"FALSE REJECTION (Genuine Rejected)\nThreshold: {threshold:.3f}", "false_reject")

def evaluate_and_plot(feature_extractor, metric_generator, val_loader, device, save_dir=None):
    """
    Main entry point function.
    """
    feature_extractor.eval()
    metric_generator.eval()
    all_scores, all_labels = [], []
    
    print("Computing distances on Validation Set...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            supports = batch['support_images'].to(device)
            queries = batch['query_images'].to(device)
            labels = batch['query_labels'].to(device)
            
            bs = supports.size(0)
            for i in range(bs):
                raw_s = feature_extractor(supports[i])
                raw_q = feature_extractor(queries[i])
                s_proto = F.normalize(raw_s, p=2, dim=1).mean(dim=0)
                q_feat = F.normalize(raw_q, p=2, dim=1)
                
                weights = metric_generator(s_proto.unsqueeze(0)).squeeze(0)
                
                dist_sq = (weights * (q_feat - s_proto).pow(2)).sum(dim=1)
                dists = torch.sqrt(torch.clamp(dist_sq, min=1e-8))
                
                all_scores.extend(dists.cpu().numpy())
                all_labels.extend(labels[i].cpu().numpy())
    
    results = compute_metrics(all_labels, all_scores)
    
    # Text Report
    print(f"\n{'='*10} FINAL METRICS {'='*10}")
    print(f"EER       : {results['eer']:.2%}")
    print(f"AUC       : {results['auc']:.4f}")
    print(f"Accuracy  : {results['accuracy']:.2%}")
    
    # Visualization Saving
    plot_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, "evaluation_metrics_dashboard.png")
        
    plot_comprehensive_evaluation(results, save_path=plot_path)
    
    return results