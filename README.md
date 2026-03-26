Language:
- English (default)
- Português: [README.pt-BR.md](README.pt-BR.md)

# Federated Learning for Brain Tumor MRI Classification with Test-Time Augmentation

This repository presents a federated learning framework for brain tumor classification using two distinct clients: one trained on **original images** and another on **preprocessed images**. The main objective is to compare local and global model performance and evaluate the impact of preprocessing and test-time augmentation (TTA) in a distributed learning setting.

---

## Associated Publication

This repository is associated with the following publication:

Exploiting Test-Time Augmentation in Federated Learning for Brain Tumor MRI Classification  
Thamara Leandra de Deus Melo, Rodrigo Moreira, Larissa Moreira, André Backes  
Proceedings of the 21st International Conference on Computer Vision Theory and Applications (VISAPP), 2026  
DOI: https://doi.org/10.5220/0014391000004084

### Citation (BibTeX)

@conference{visapp26,
author={Thamara Melo and Rodrigo Moreira and Larissa Moreira and André Backes},
title={Exploiting Test-Time Augmentation in Federated Learning for Brain Tumor MRI Classification},
booktitle={Proceedings of the 21st International Conference on Computer Vision Theory and Applications - Volume 1: VISAPP},
year={2026},
pages={148-156},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0014391000004084},
isbn={978-989-758-804-4},
}

---

## Preprocessing

Image preprocessing is performed by a dedicated script preprocess.py, responsible for:

* Resizing (with or without aspect ratio preservation)
* Optional grayscale conversion
* RGB enforcement for compatibility with architectures such as ResNet18
* Noise reduction filters (Gaussian/Bilateral)
* Optional contrast enhancement using CLAHE
* Format standardization (JPG/PNG)
* Dataset reorganization by class

This step is executed prior to federated training and is applied exclusively to the preprocessed client.

---

## Project Objective

Develop and evaluate a brain tumor classification model using Federated Learning (FL) with two clients:

* Client 1: Original MRI images (Kaggle dataset)
* Client 2: Preprocessed images with:

  * Resizing
  * Grayscale conversion
  * Normalization
  * Smoothing filters
  * Histogram equalization

The project aims to answer:

* Does preprocessing improve local model performance?
* Is the federated model more robust than individual models?
* How does FL behave with heterogeneous data?

---

## System Architecture

The experiment uses a FedAvg-based architecture:

* Server: Aggregates model weights from clients
* Clients (2):

  * Original client
  * Preprocessed client
* Model: ResNet18

Each client trains locally and sends weights to the server, which aggregates and redistributes them.

---

## Requirements

* Python 3.8+
* PyTorch
* NumPy
* Matplotlib
* Flower (or another FL framework)
* Scikit-image / OpenCV

---

## How to Run

### 1. Prepare the dataset

data/original/  
data/preprocessed/

### 2. Start the server

python federated/server.py

### 3. Start the clients

python -c "from federated.client import start_client; start_client('dataset_kaggle/Training')"  
python -c "from federated.client import start_client; start_client('dataset_kaggle_preprocessed/Train')"

---

## Test-Time Augmentation (TTA)

After federated training, the global model is evaluated using TTA (Test-Time Augmentation), improving prediction robustness.

Transformations include:

* Rotation
* Horizontal/vertical flip
* Light perturbations (noise, brightness variation)

Process:

1. Generate multiple variations of each test image
2. Run inference for each variation
3. Aggregate predictions (mean or majority voting)

---

## Expected Results

The study allows comparison of:

* Local model accuracy
* Federated model accuracy
* Convergence behavior
* Impact of preprocessing and TTA

---

## Final Remarks

This repository was developed for academic purposes, enabling controlled experiments with heterogeneous data in a federated learning scenario.
