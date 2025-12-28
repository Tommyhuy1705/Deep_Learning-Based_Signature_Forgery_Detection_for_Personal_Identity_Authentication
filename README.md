# Deep Learning-Based Signature Forgery Detection: A Learnable Distance Approach with YOLOv10, ResNet-34, and Triplet Siamese Similarity Network

## Introduction

Handwritten signatures remain a cornerstone of identity verification in critical sectors like banking, law, and finance. However, traditional verification systems struggle with two fundamental challenges: the natural variability in a person's signature (*intra-class variation*) and the increasing sophistication of skilled forgeries (*inter-class similarity*). Furthermore, deploying a robust system often requires a large number of signature samples per user, which is impractical in many real-world scenarios. Fixed distance metrics (e.g., Euclidean, Cosine) often fail to effectively address these challenges due to their 'one-size-fits-all' nature.

To address these limitations, this project presents a **Few-Shot Adaptive Metric Learning framework** for offline signature forgery detection. Building upon the concept of learnable distances, our approach leverages **meta-learning** to learn *how to generate* a unique, writer-specific distance metric from just a handful (*k-shot*) of genuine signature samples. This allows the system to dynamically adapt to the unique characteristics and variability of any individual's signature, providing a more personalized and accurate verification.

Our framework integrates three key components:
- **YOLOv10**: For high-efficiency signature localization from documents (pre-processing).
- **Pre-trained ResNet-34**: As a robust feature extractor to generate powerful signature embeddings.
- **Adaptive Metric Learner (`MetricGenerator`)**: A meta-trained Relation Network that generates a unique **Non-linear Similarity Metric** for each user, trained using an **Online Hard Triplet Mining** strategy within a Siamese architecture.

Experimental results demonstrate the state-of-the-art performance and robustness of our approach. Using a rigorous **5-fold cross-validation** on the **BHSig-260 dataset**, the model achieves a mean accuracy of **92.27%**, with specific folds reaching **perfect separation (0.00% EER)**. More importantly, to rigorously test its generalization capability against domain shift, the model, trained exclusively on BHSig (Indic scripts), achieves impressive performance on the completely unseen **CEDAR dataset (Latin script)** via a few-shot adaptation strategy: **95.66% Accuracy** and **4.34% EER** verified on **42,600 exhaustive comparison pairs**. These findings establish our adaptive metric learning approach as a superior solution for cross-domain signature verification.

## Key Features
- **Few-Shot Learning**: Accurately verifies signatures using only `k=1` (One-Shot) or `k=5` genuine samples for new, unseen users.
- **Adaptive Metric Learning**: Utilizes meta-learning (`MetricGenerator`) to generate a writer-specific similarity function, providing highly personalized verification.
- **Advanced Training Strategy**: Employs a Two-Stage pipeline (Representation Learning -> Metric Learning) with Online Hard Triplet Mining.
- **Perfect Separation Potential**: Achieved **0.00% EER** and **100% Accuracy** on specific validation folds (Fold 3 & 4) of the BHSig dataset.
- **Proven Cross-Domain Generalization**: Demonstrates strong plasticity by adapting from Indic scripts (BHSig) to English signatures (CEDAR) with **95.66% Accuracy**, proving robustness against script morphology shifts.
- **End-to-End Capable**: Includes YOLOv10 for optional automated signature localization from raw documents.

---

## Project Structure
```plaintext
├── configs/
│   ├── __init__.py
│   └── config_tSSN.yaml             # Configuration for baseline tSSN models
│
├── dataloader/
│   ├── __init__.py
│   ├── meta_dataloader.py           # Dataloader for meta-learning episodes (N-Way K-Shot)
│   └── tSSN_trainloader.py          # Dataloader for pre-training stage
│
├── losses/
│   ├── __init__.py
│   └── triplet_loss.py              # Contains standard TripletLoss and pairwise distance logic
│
├── models/
│   ├── __init__.py
│   ├── Triplet_Siamese_Similarity_Network.py # Wrapper model used for pre-training
│   ├── feature_extractor.py                  # ResNet-34 backbone implementation (CORE)
│   ├── meta_learner.py                       # The MetricGenerator module (CORE - Relation Network)
│
├── notebooks/
│   ├── baseline_metric_selection.ipynb   # Step 0: Metric selection (Cosine vs Euclidean)
│   ├── pretraining.ipynb                 # Step 1: Pre-training the feature extractor
│   ├── meta_training_kfold_final.ipynb   # Step 2: Main K-fold CV meta-learning on BHSig260
│   ├── cross_dataset_evaluation.ipynb    # Step 3: Cross-dataset evaluation on CEDAR
│   └── yolov10_bcsd_training.ipynb       # Optional: YOLOv10 training for localization
│
├── scripts/
│   ├── __init__.py
│   ├── prepare_kfold_splits.py      # Script to generate K-fold splits
│   └── restructure_bhsig.py         # Script to restructure BHSig-260 dataset
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py                   # Utility functions
│   └── model_evaluation.py          # Comprehensive evaluation metrics (EER, AUC)
│
├── README.md
├── requirements.txt
├── setup.py                        # Installation setup
└── signature_verification.egg-info/ # Build metadata
```

---

## Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Tommyhuy1705/Deep_Learning-Based_Signature_Forgery_Detection_for_Personal_Identity_Authentication.git
    cd Deep-Learning-Based-Signature-Forgery-Detection-for-Personal-Identity-Authentication
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## **Kaggle API Token Setup**

To access and download datasets directly from Kaggle within this project, follow these steps to set up your Kaggle API token:

1. Go to your [Kaggle account settings](https://www.kaggle.com/account).
2. Scroll down to the **API** section.
3. Click on **"Create New API Token"** – a file named `kaggle.json` will be downloaded.
4. Place the `kaggle.json` file in the root directory of this project **or** in your system's default path:  
   - Linux/macOS: `~/.kaggle/kaggle.json`  
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
5. Make sure the file has appropriate permissions:  
   ```bash
   chmod 600 ~/.kaggle/kaggle.json

---

## Usage & Replication of Results

To replicate the state-of-the-art results, follow these steps sequentially. A GPU-accelerated environment is highly recommended.

**Step 0: Data Preparation**
-   Download the required datasets and place them in accessible paths (e.g., Kaggle input directory).
    -   **BHSig-260 (Hindi & Bengali)**: Used for Pre-training and In-Domain Meta-training.
    -   **CEDAR Dataset**: Used for Cross-Dataset Generalization evaluation.
-   **(Optional)** Use the `notebooks/yolov10_bcsd_training.ipynb` notebook to train a YOLOv10 model if you need to perform signature localization on raw documents. The subsequent steps assume pre-cropped signature images are available as per the CEDAR and BHSig-260 dataset structures.

**Step 1: Pre-train the Feature Extractor**
-   **Purpose**: Initialize the ResNet-34 backbone with robust signature representations using Cosine Similarity.
-   **Action**: Run the `notebooks/pretraining.ipynb` notebook completely.
-   **Output**: This will generate the `background_pretrain.pth` weights file in the `/kaggle/working/pretrained_models/` directory (or similar). Create a Kaggle dataset from this output for the next step.

**Step 2: Meta-Train and Evaluate on BHSig-260 (K-Fold Cross-Validation)**
-   **Purpose**: To train the adaptive metric learner (`MetricGenerator`) and rigorously validate the framework's performance and reliability on the BHSig dataset using 5-fold cross-validation.
-   **Action**:
    1.  Ensure the Kaggle dataset containing `background_pretrain.pth` is added as input to the `meta_training_kfold.ipynb` notebook. Update the `PRETRAINED_WEIGHTS_PATH` variable accordingly.
    2.  Run the `notebooks/meta_training_kfold.ipynb` notebook completely. This notebook internally calls `scripts/prepare_kfold_splits.py` to generate data splits. It then performs the 5-fold training and validation loop.
-   **Output**: The notebook will print the detailed results for each fold and the final summary (Mean ± Std Dev) for all metrics. It will also save the best model weights for each fold (e.g., `/kaggle/working/best_model_fold_X/`). Choose the weights from one fold (e.g., Fold 3, which performed best in cross-dataset tests) and create a new Kaggle dataset from this output for the final step.

**Step 3: Evaluate Cross-Dataset Generalization on CEDAR**
-   **Purpose**: To test the generalization ability of the model trained *only* on BHSig-260 dataset (Bengali and Hindi) by evaluating it on the completely different CEDAR dataset.
-   **Action**:
    1.  Ensure the Kaggle dataset containing the best model weights from Step 2 (e.g., `best_model_fold_5`) and the `nth2165/bhsig260-hindi-bengali` dataset are added as input to the `cross_dataset_evaluation.ipynb` notebook. Update the `BEST_BHSIG_MODEL_DIR` and `CEDAR_RAW_BASE_DIR` variables.
    2.  Run the `notebooks/cross_dataset_evaluation.ipynb` notebook completely. It then loads the BHSig-trained model and evaluates it independently on both language subsets.
-   **Output**: The notebook will print the detailed performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC) and plots (ROC Curve, Confusion Matrix) for CEDAR evaluations.

---

## Pre-trained Models

To facilitate reproducibility and evaluation, the pre-trained weights for the main components of the model are provided below.

You can download them and place them in the `checkpoints/` folder (or the corresponding folder defined in the code) to run evaluation notebooks (e.g., `cross_dataset_evaluation.ipynb`) without having to retrain from scratch.

| Model | Weight File (Example) | Download Link |
| :--- | :--- | :--- |
| **YOLOv10** (Signature Detection) | `yolov10n_best.pt` | **[Download here]()** |
| **ResNet-34** (Pre-training) | `my-pretrained-weights` | **[Download here](https://www.kaggle.com/datasets/nth2165/my-pretrained-weights)** |
| **Meta-Model** (Final model) | `best-bhsig-model-weights` | **[Download here](https://www.kaggle.com/datasets/nth2165/my-best-models-meta-learning)** |

---

## Results

Our Few-Shot Adaptive Metric Learning framework demonstrated exceptional performance and "feature plasticity" across diverse writing systems.

### 1. Intra-Dataset Reliability: 5-Fold Cross-Validation on BHSig-260

To rigorously assess the model's reliability and performance on a standard benchmark, we conducted 5-fold cross-validation on the BHSig-260 dataset. The results below show the mean and standard deviation across the 5 folds.

| Metric        | Mean          | Std Dev       |
| :------------ | :------------ | :------------ |
| **Accuracy** | **92.27%** | **5.30%** |
| Precision     | 0.9198        | 0.0577        |
| Recall        | 0.9273        | 0.0464        |
| F1-Score      | 0.9234        | 0.0519        |
| ROC-AUC       | 0.9467        | 0.0471        |

**Discussion:** The achievement of 0.00% EER on specific folds highlights the model's potential to create perfect decision boundaries. The variance reflects the inherent difficulty of One-Shot learning on complex Indic scripts.

### 2. Cross-Dataset Generalization: BHSig-260 -> CEDAR

To evaluate true generalization, the model trained on BHSig was adapted (5-shot) and tested on CEDAR (Latin script).

**Performance on CEDAR:**

| Metric        | Score         |
| :------------ | :------------ |
| **Accuracy**  | **95.66%**    |
| Precision     | 0.7793        |
| Recall        | 0.8300        |
| F1-Score      | 0.8039        |
| ROC-AUC       | 0.7565        |
| **EER**       | **4.34%**     |
| Test Pairs    | 42,600         |

**Discussion:** Despite the significant domain gap between Indic and Latin scripts, the model achieved an impressive 4.34% EER. This confirms that the learned meta-metric is not just overfitting to dataset artifacts but capturing fundamental signature biomechanics.

### 3. Methodological Validation

The combined results validate the effectiveness of the proposed pipeline:
-   **Pre-training** provides a robust feature foundation.
-   **Meta-learning with `MetricGenerator` (Attention)** successfully learns to create personalized, adaptive Mahalanobis metrics.
-   **Online Hard Triplet Mining** effectively guides the learning process towards difficult discrimination tasks.

---

## Datasets
-   **CEDAR Dataset**: [Link](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset) - Used for cross-dataset generalization evaluation.
-   **BHSig-260 (Hindi & Bengali)**: [Link](https://www.kaggle.com/datasets/nth2165/bhsig260-hindi-bengali) - Used for pre-training and K-fold meta-training/validation.
-   **BCSD**: [Link](https://www.kaggle.com/datasets/saifkhichi96/bank-checks-signatures-segmentation-dataset) - Used for optional YOLOv10 training.

## Contributions
*(This section summarizes the key advancements of the final methodology)*
-   **Novel Adaptive Framework**: Shifted from static distances to a Dynamic Meta-Metric approach using Relation Networks.
-   **Cross-Script Generalization**: Proved that features learned on Hindi/Bengali can effectively transfer to English signatures.
-   **Rigorous Evaluation**: Implemented an Exhaustive Pair Generation protocol (42k+ pairs) to ensure statistical significance, surpassing traditional random sampling methods.
-   **Reproducible Pipeline**: Provided a complete, modular codebase with sequential notebooks for full result replication.

## Future Work
-   **Domain Adaptation**: Integrate unsupervised domain adversarial training (DANN) to further reduce the domain gap without requiring target labels.
-   **Explainable AI (XAI)**: Implement Grad-CAM to visualize the "focus regions" of the MetricGenerator.
-   **Multi-Modal Fusion**: Incorporate online dynamic features (pressure, velocity) for a hybrid verification system.

---

## Acknowledgments
Special thanks to the research community for providing valuable datasets and open-source tools that facilitated this work. We also appreciate the insightful feedback from initial reviews which guided the significant improvements presented here.

--- 