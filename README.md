# Deep Learning-Based_Signature Forgery Detection for Personal Identity Authentication
## Introduction  

Handwritten signatures continue to serve as a widely accepted form of identity verification across domains such as banking, legal documentation, and governmental services. However, the increasing sophistication of forgery techniques presents serious challenges to the reliability of traditional verification systems, which are often rule-based or reliant on handcrafted features.

To address these limitations, this project presents a deep learning-based framework for offline signature forgery detection, leveraging a Triplet Siamese Similarity Network (tSSN) trained with triplet loss. The proposed system integrates three key components:
- **YOLOv10**: efficient signature localization from scanned document images.
- **ResNet-34**: the feature extractor to generate robust, high-dimensional embeddings of signature images.
- **Triplet Network with Triplet Loss**: learn a discriminative embedding space that enforces minimal distance between genuine signature pairs and maximal distance from forgeries.

A novel contribution of this work is the integration of multiple distance metrics—including Euclidean, Cosine, Manhattan, and a learnable distance function—to investigate how similarity definitions affect verification performance. Experimental results show that using Euclidean distance with a margin of 0.6 achieves the highest accuracy of 95.6439% on the CEDAR dataset, significantly outperforming previous benchmarks.

The system is trained using balanced batch sampling, enabling dynamic construction of hard and semi-hard triplets during training and improving model generalization across diverse handwriting styles. Evaluation metrics include accuracy, precision, recall, ROC-AUC, FAR, FRR, and EER.

This project offers a scalable, accurate, and generalizable solution for signature-based identity authentication, with direct applicability in high-security environments such as banking, finance, and legal processes.

## **Features**
- Offline signature forgery detection based on deep metric learning.
- Signature region localization using YOLOv10.
- Embedding extraction via ResNet-34 backbone.
- Metric learning with Triplet Loss using four distance modes:
 - **Euclidean distance**
 - **Cosine distance**
 - **Manhattan distance**
 - **Learnable distance**
- Evaluation with accuracy, ROC-AUC, EER, precision, recall.
- Experimental margin tuning: [0.2, 0.4, 0.6, 0.8, 1.0].
- Balanced batch sampling for consistent triplet generation.

---

## Project Structure  
```plaintext
├── configs/                         # Configuration files
│   ├── __init__.py
│   └── config_tSSN.yaml             # Model and training hyperparameters
│
├── dataloader/                     # Custom data loading and triplet construction
│   ├── __init__.py
│   └── tSSN_trainloader.py         # Triplet loader and balanced batch sampler
│
├── losses/                         # Triplet loss and metric logic
│   ├── __init__.py
│   └── triplet_loss.py             # Supports Euclidean, Cosine, Manhattan, Learnable
│
├── models/                         # Model definitions
│   ├── __init__.py
│   ├── Triplet_Siamese_Similarity_Network.py  # Main tSSN architecture
│   └── feature_extractor.py        # ResNet-34 embedding extractor
│
├── notebooks/                      # Jupyter notebooks for development and visualization
│   ├── final_evaluation.ipynb
│   ├── model_training.ipynb
│   └── yolov10-bcsd_training.ipynb
│
├── utils/                          # Helper scripts and evaluation tools
│   ├── __init__.py
│   ├── helpers.py                  # Miscellaneous utilities
│   └── model_evaluation.py         # ROC, accuracy, precision, etc.
│
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── setup.py                        # Installation setup (optional for packaging)
├── signature_verification.egg-info/  # Build metadata (auto-generated)
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   ├── dependency_links.txt
│   ├── requires.txt
│   └── top_level.txt
```

---

## **Installation**
Follow the steps below to set up the project:

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/Tommyhuy1705/Deep_Learning-Based_Signature_Forgery_Detection_for_Personal_Identity_Authentication.git
   cd your-repo-name
   ```

2. **Install dependencies**:  
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

## **Usage**
- To train and evaluate the model, follow these steps:
**Step 1: Configure training parameters**
  ```Edit the config.yaml file under configs/ to set:
     Model backbone (ResNet34)
     Feature embedding dimension
     Margin value and distance mode (euclidean, cosine, etc.)
     Batch size, learning rate, number of epochs
  ```
**Step 2: Localize signatures using YOLOv10**
  ```Run the notebook:
     notebooks/yolov10-bcsd_training.ipynb
  ```
**Step 3: Train the model**
  ```Run the notebook:
     notebooks/model_training.ipynb
  ```
**Step 4: Evaluate performance**
  ```Open and run the notebook:
     notebooks/final_evaluation.ipynb
  ```
---

## **Results**
### Key Findings:
1. **Best-performing configuration:**
- Triplet Network with Euclidean distance and margin = 0.6
- Accuracy: 95.6439% on CEDAR dataset
2. Learnable distance function showed potential but did not outperform fixed metrics.
3. Balanced batch sampling improved generalization across user styles.
4. Embedding visualizations show clear separation between genuine and forged signatures.

___
 
---

## **Contributions**

- Designed and implemented the full pipeline for offline signature verification using a Triplet Siamese Network (tSSN).
- Integrated YOLOv10 for efficient signature region localization from scanned documents.
- Developed flexible Triplet Loss module supporting multiple distance metrics: Euclidean, Cosine, Manhattan, and Learnable.
- Implemented a balanced batch sampler to improve triplet selection and training stability.
- Conducted extensive experiments with margin tuning and distance metric variations.
- Achieved 95.6439% accuracy on the CEDAR dataset using Euclidean distance with margin = 0.6.
- Visualized performance through ROC curves, precision-recall metrics, and embedding space analysis.
- Structured the project for reproducibility and scalability, using modular PyTorch components and well-documented notebooks.
- Prepared supporting materials including dataset configuration, training logs, and evaluation tools.

---

## **Future Work**
- Cross-dataset evaluation on GPDS, BHSig260 for generalizability.
- Integrate lighter backbones (e.g., MobileNet) for real-time performance.
- Incorporate attention mechanisms for enhanced local feature focus.
- Explore adaptive or learnable margin strategies.
- Apply to multilingual and multicultural signature styles.
- Introduce explainable AI components for visualizing decision-making process.

---

## **Acknowledgments**
Special thanks to the contributors and open-source community for providing tools and resources.

--- 


