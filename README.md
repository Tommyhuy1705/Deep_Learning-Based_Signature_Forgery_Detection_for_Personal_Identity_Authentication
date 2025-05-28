# Deep Learning-Based_Signature Forgery Detection for Personal Identity Authentication
## Introduction  

Handwritten signatures continue to serve as a widely accepted form of identity verification across domains such as banking, legal documentation, and governmental services. However, the increasing sophistication of forgery techniques presents serious challenges to the reliability of traditional verification systems, which are often rule-based or reliant on handcrafted features.

To address these limitations, this project presents a deep learning-based framework for offline signature forgery detection, leveraging a Triplet Siamese Similarity Network (tSSN) trained with triplet loss. The proposed system integrates three key components:

YOLOv10 for efficient signature localization from scanned document images.

ResNet-34 as the feature extractor to generate robust, high-dimensional embeddings of signature images.

Triplet Network with Triplet Loss to learn a discriminative embedding space that enforces minimal distance between genuine signature pairs and maximal distance from forgeries.

A novel contribution of this work is the integration of multiple distance metrics—including Euclidean, Cosine, Manhattan, and a learnable distance function—to investigate how similarity definitions affect verification performance. Experimental results show that using Euclidean distance with a margin of 0.6 achieves the highest accuracy of 95.6439% on the CEDAR dataset, significantly outperforming previous benchmarks.

The system is trained using balanced batch sampling, enabling dynamic construction of hard and semi-hard triplets during training and improving model generalization across diverse handwriting styles. Evaluation metrics include accuracy, precision, recall, ROC-AUC, FAR, FRR, and EER.

This project offers a scalable, accurate, and generalizable solution for signature-based identity authentication, with direct applicability in high-security environments such as banking, finance, and legal processes.

## **Features**
___

---

## Project Structure  
```plaintext
___
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
___

---

## **Results**
### Key Findings:
1. ****:

___
 
---

## **Contributions**
___

---

## **Future Work**
___

---

## **Acknowledgments**
Special thanks to the contributors and open-source community for providing tools and resources.

--- 


