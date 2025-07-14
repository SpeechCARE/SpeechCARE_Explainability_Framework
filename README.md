# Explainability Framework

The **Explainability Framework** is a comprehensive system designed to deliver accurate, interpretable, and human-understandable explanations of a model's output regarding a patient's cognitive status based on specific inputs. The framework is divided into two main modules:

---

## 1. Patient Health Assessment

This module aims to provide a deep understanding of the patient's **clinical**, **functional**, and **social** status. It includes three main components:

- **Clinical and Functional Overview**
- **Lab Tests**
- **Social Determinants of Health (SDoH) and Clinical Reports**

In this module, we collaborated closely with doctors and specialists to identify the most reliable and direct indicators of cognitive impairments, as well as the threshold values that signify risk. Each patient is assessed according to these factors, and any signs of cognitive risk—based on those critical values—are flagged and reported accordingly.

---

## 2. Speech Explainability

This module consists of two sections:

### Linguistic Module

We provide detailed interpretations of speech transcripts using **SHAP (SHapley Additive exPlanations)** in conjunction with a set of **extracted linguistic features**. This allows for a clearer understanding of how different language characteristics contribute to the model's output. (More details in `Explainability_Linguistic.ipynb`)

### Acoustic Module

We offer in-depth explanations of the speaker's audio using techniques such as:

- Saliency Mapping
- SHAP
- Analysis of the audio signal using features such as:
  - Informative and non-informative pauses
  - Fundamental frequencies (F0)
  - Third formant frequency (F3)
  - Rhythmicity and monotonicity (via _Shannon Entropy_)
  - Energy in the frequency domain
  - Shimmer standard deviation

These features and their interpretations are visualized over **raw audio waveforms** and **spectrograms**. We also provide **plots of normalized Shannon entropy** to illustrate speech complexity. (More details in `Explainability_Acoustic.ipynb`)

---

## 📘 Tutorial

This repository includes a **Tutorial** section, where we demonstrate—**in detail and with numerous examples**—how to interpret and read a spectrogram. (More details in `Spectrogram_Tutorial.ipynb`)

---

## 📁 Repository Structure

```
├── acoustic_module/         # Audio-based feature extraction and explainability tools
├── data/                    # Raw and preprocessed input data
├── dataset/                 # Dataset loading and preprocessing scripts
├── explainability/          # Explainability core modules
│   ├── Gradient_based/      # Gradient-based interpretability methods
│   ├── plotting/            # Plotting utilities for explanations
│   ├── SHAP/                # SHAP-based explanation methods
│   ├── tutorial/            # Spectrogram interpretation tutorials
├── interface/               # User interface components (e.g., for demo or app)
├── interpretation/          # Final output generation and integration logic
├── linguistic_module/       # Text-based feature extraction and interpretation
├── model/                   # Model architectures and training scripts
├── utils/                   # Utility functions and shared helpers

```
