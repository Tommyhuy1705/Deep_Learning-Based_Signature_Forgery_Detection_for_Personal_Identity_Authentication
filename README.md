# Deep_Learning-Based_Signature_Forgery_Detection_for_Personal_Identity_Authentication
## Introduction  

Handwritten signature remains one of the most widely used biometric modalities for identity verification in various real-world applications, particularly in secure domains such as banking, legal document authentication, and personal identification. However, signature forgery — both skilled and unskilled — poses a significant threat to the reliability of such systems. Traditional approaches often struggle to generalize well across diverse writing styles and varying levels of forgery.
This project explores deep learning-based approaches to address the problem of offline signature verification and forgery detection. Specifically, we investigate and compare three powerful models:

- Siamese Network + Contrastive Loss + ResNet50: A pairwise metric learning model that learns to distinguish genuine and forged signatures by minimizing the distance between similar pairs and maximizing it for dissimilar ones.
- Triplet Network + Triplet Loss + ResNet50: A relative distance learning framework that pushes genuine signature pairs closer while pushing forgeries further apart using anchor-positive-negative triplets.
- ResNet50 + ArcFace Loss: A classification-based approach with angular margin loss that enhances the discriminative power of signature embeddings in high-dimensional space, inspired by state-of-the-art face recognition methods.

All models are trained and tested on public benchmark datasets, and their performance is compared based on verification accuracy, forgery detection rate, generalization to unseen users, and embedding visualization.

The goal of this research is to develop robust, scalable, and intelligent verification systems that can be reliably deployed in real-world identity authentication workflows, minimizing the risk of signature-based fraud.
---

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
   git clone https://github.com/Tommyhuy1705/Mean_Shift_Clustering.git
   cd your-repo-name
   ```

2. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**
___

---

## **Results**
### Key Findings:
1. **Mean Shift Clustering**:

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


