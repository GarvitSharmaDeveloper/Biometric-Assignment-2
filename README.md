# CS 228 - Biometric Security with AI: Assignment 2
## Multiclass Clean-Label Data Poisoning on CIFAR-10

This repository contains the implementation of a clean-label data poisoning attack on a 4-class subset of the CIFAR-10 dataset, following the "Poison Frogs" methodology by Shafahi et al. (2018).

### 🚀 Overview
The goal of this project is to perform a targeted attack where "poison" images are inserted into a model's training set. These images are visually indistinguishable from a **base class** (Automobile) but are optimized to collide in the feature space of a specific **target image** (Airplane). After retraining, the model misclassifies the target image as the base class, while maintaining high overall accuracy.

### 📁 Project Structure
- `final_assignment2.ipynb`: Complete Jupyter Notebook containing data preparation, model architecture, poison generation, retraining, and evaluation.
- `Assignment 2 Report.pdf`: A comprehensive technical report summarizing the methodology, results (10% attack success rate), and visual analysis of the poison evolution.
- `assignment2.py`: Python script version of the core implementation.

### 🛠️ Methodology
1.  **Data Preparation**: Extracted a 4-class subset (Airplane, Automobile, Bird, Cat) from CIFAR-10 with 500 images per class.
2.  **Model Architecture**: Built a custom 3-layer Convolutional Neural Network (CNN) with a 256-dimensional penultimate feature layer.
3.  **Poison Generation**: Used an iterative optimization loop to minimize the Euclidean distance between the poison's features and the target's features, while maintaining visual similarity to the base image using a Frobenius norm constraint ($\beta = 0.05$).
4.  **Evaluation**: Retrained a fresh model from scratch on the poisoned dataset and measured the Attack Success Rate (ASR) on held-out target images.

### 📊 Results
- **Clean Model Accuracy**: ~72.83%
- **Poisoned Model Accuracy**: ~70.95%
- **Attack Success Rate**: **10.00%** (1 out of 10 targets successfully misclassified as the base class).
- **Stealth**: Overall model performance remained high, and poison images remained visually identical to the base class.

### ⚙️ How to Run
1.  Open `final_assignment2.ipynb` in Google Colab or a local Jupyter environment.
2.  Ensure `torch`, `torchvision`, and `matplotlib` are installed.
3.  Run all cells to replicate the poison generation and evaluation process.

### 📝 Author
**Garvit Sharma**
CS 228 - Biometric Security with AI
May 6, 2026
