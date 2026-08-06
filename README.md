# A Representation-Aware Late Fusion Framework for Generalizable Atrial Fibrillation Detection From Single-Lead ECG

## 📖 Overview
This repository contains the implementation of the project: **"A Representation-Aware Late Fusion Framework for Generalizable Atrial Fibrillation Detection From Single-Lead ECG"**. 

Cardiovascular diseases (CVDs) remain a leading cause of global mortality. The continuous monitoring required to detect sporadic paroxysmal arrhythmias is effectively achieved through single-lead Electrocardiograms (ECGs) integrated into wearable Internet of Medical Things (IoMT) devices. However, Atrial Fibrillation (AF)—a severe arrhythmia posing high stroke risks—is highly unstable, multimodal, and exhibits chaotic electrophysiological characteristics. 

Historically, automated detection relied on 1D Convolutional Neural Networks (CNNs) analyzing temporal representations, which struggle to capture the complex spectral properties of fibrillary waves and exhibit poor generalization across diverse clinical datasets. To address this, this framework analyzes physiological data across three distinct domains: Time (1D Raw ECG), Time-Frequency (STFT), and Scale-Time (CWT). The resulting probabilistic inferences are dynamically weighted using a representation-aware late fusion Multi-Layer Perceptron (MLP) to ensure robust generalization against hardware-induced domain shifts.

## 🫀 Clinical Background & Challenges
A normal cardiac cycle operates with a synchronized electrical sequence, characterized by the P-QRS-T complex. 

![Figure 2.2: Morphology of a healthy person’s ECG wave, showing the P-QRS-T wave pattern, normal physiological amplitudes, and isoelectric intervals.](path/to/figure2.2.png)

In contrast, AF is caused by multiple spontaneous ectopic firings, leading to chaotic electrical disarray. In a single-lead ECG, this manifests as the absence of distinct P waves, the presence of continuous fibrillatory (f) waves, and highly irregular R-R intervals. 

![Figure 2.3: ECG morphology comparison of single lead recording: (a) Normal Sinus Rhythm (NSR) and (b) An episode of Atrial Fibrillation (AF).](path/to/figure2.3.png)

Single-lead ECGs acquired from ambulatory devices are inherently noisy, susceptible to baseline wander, muscle artifacts, and structural variability. 

## 🧠 Proposed Architecture & Methodology
The intelligent arrhythmia detection system is designed as an end-to-end pipeline that transforms raw 1D signals into multi-dimensional multimodal representations. 

![Figure 5.1: Proposed research methodology end-to-end workflow showing data preprocessing, three parallel neural networks pipeline, and late fusion approach.](path/to/figure5.1.png)

### 1. Data Preprocessing & Windowing
To maintain mathematical consistency across diverse datasets, all continuous ECG recordings were systematically resampled to 300 Hz. The data was segmented into fixed-length windows of precisely 9,000 discrete samples. Static Z-score normalization was applied using parameters derived solely from the primary training set.

![Figure 5.2: Length distribution of the ECG signal segments in the PhysioNet data set.](path/to/figure5.2.png)

### 2. Multimodal ECG Representations & Base Models
The 9000x1 ECG vector is transformed into three complementary feature spaces.

![Figure 5.3: NSR (left) and AF (right) multi-domain comparison across Raw, FFT, STFT, and CWT.](path/to/figure5.3.png)

*   **Time-Domain (Raw ECG)**: Captures the time-domain morphology to map sequential rhythmic irregularities. Processed using an optimized 1D-RNN architecture with L2 weight decay and aggressive dropout to prevent overfitting.

| Layer | Kernel | Stride | Output Shape | Params |
| :--- | :--- | :--- | :--- | :--- |
| Input | - | - | (9000,1) | 0 |
| Conv1D (32) | 7 | 2 | (4500,32) | 256 |
| BatchNorm | - | - | (4500,32) | 128 |
| MaxPool | 2 | 2 | (2250,32) | 0 |
| Conv1D (64) | 5 | 2 | (1125,64) | 10304 |
| ... | ... | ... | ... | ... |
| Dense (1) | - | - | (1) | 129 |
*Table 5.1: Architectural Parameters of the Proposed 1D-RNN Model (Raw ECG)*

*   **Time-Frequency Domain (STFT)**: Reveals hidden high-frequency spectral variations. The signal is transformed into 2D spectrograms using Short-Time Fourier Transform (STFT). Evaluated using a highly parameterized ResNet50 model.

| Layer | Kernel | Stride | Output Shape | Params |
| :--- | :--- | :--- | :--- | :--- |
| Input | - | - | (129,283,1) | 0 |
| Conv2D (16) | 3x3 | 1 | (129,283,16) | 160 |
| ResNet50 Backbone | - | - | (4,9,2048) | 23,587,712 |
| Dense (2) | - | - | (2) | 1,026 |
*Table 5.2: Architectural Parameters of the ResNet50 Model (STFT-Based)*

*   **Scale-Time Domain (CWT)**: Utilizes the Continuous Wavelet Transform (CWT) for multi-resolution localization—high temporal resolution for QRS peaks and high frequency resolution for f-waves. Handled by a specialized U-Net encoder utilizing symmetric spatial down-sampling.

| Layer | Kernel | Stride | Output Shape | Params |
| :--- | :--- | :--- | :--- | :--- |
| Input | - | - | (64,4500,1) | 0 |
| Conv2D (32) | 3x3 | 1 | (64,4500,32) | 320 |
| MaxPool | (2,4) | (2,4) | (32,1125,32) | 0 |
| Dense (2) | - | - | (2) | 514 |
*Table 5.3: Architectural Parameters of the CWT-Based CNN Encoder*

### 3. Representation-Aware Late Fusion
The framework extracts the individual probability vectors ($P_{raw}$, $P_{stft}$, $P_{cwt}$) from the base models. These probabilities are concatenated and passed through a lightweight MLP diagnostic aggregation tool to compute the optimal weighting for a conclusive decision.

![Figure 5.4: Architecture of Late-Fusion Framework in Detail.](path/to/figure5.4.png)

| Layer (Type) | Output Shape | Param # |
| :--- | :--- | :--- |
| InputLayer | (None, 3) | 0 |
| Dense (16) | (None, 16) | 64 |
| Dropout | (None, 16) | 0 |
| Dense (8) | (None, 8) | 136 |
| Dropout | (None, 8) | 0 |
| Dense (1) | (None, 1) | 9 |
*Table 5.4: Layer-wise architectural schematic of the MLP utilized for late fusion aggregation.*

## 📂 Datasets
This research systematically uses two fundamentally different datasets to strictly test domain generalization without employing synthetic data augmentation.

| Feature | MIT-BIH AF Database (External Validation) | PhysioNet/CinC 2017 (Primary Training) |
| :--- | :--- | :--- |
| Origin/Device | Ambulatory Holter monitors | AliveCor handheld device |
| Duration | 10 hours per record | 9-60 seconds |
| Sampling Rate | 250 Hz | 300 Hz |
| Resolution | 12-bit (range: ±10 mV) | Not specified |
*Table 4.1: Detailed Comparison of MIT-BIH AF and PhysioNet/CinC 2017 Datasets*

## 🚀 Performance & Results
### Internal Validation (PhysioNet Dataset)
The Late Fusion model dynamically mitigates the weaknesses of individual representations, producing the following internal training metrics. 

| Representation | Accuracy (%) | Recall (%) | Specificity (%) | Precision (%) | F1-score (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RNN (Raw ECG) | 89.16 | 91.36 | 76.16 | 95.78 | 93.51 |
| ResNet50 (STFT) | 95.21 | 95.38 | 94.19 | 98.98 | 97.15 |
| U-Net (CWT) | 89.83 | 95.09 | 58.72 | 93.17 | 94.12 |
| **Late Fusion (Threshold=0.51)** | **97.06** | **97.54** | **94.19** | **98.91** | **98.27** |
*Table 6.1: Performance (training metrics) of the models selected from each category*

![Figure 6.8: Bar graph showing the comparison between the proposed multi-modal Fusion Model and the baseline models.](path/to/figure6.8.png)
![Figure 6.9: Confusion Matrix for Representation-Aware Late Fusion approach.](path/to/figure6.9.png)
![Figure 6.10: Comparison of AUC-ROC curves.](path/to/figure6.10.png)

### External Cross-Dataset Validation (MIT-BIH AF Dataset)
To rigorously evaluate real-world resilience, the frozen model was deployed on the unseen MIT-BIH AF dataset with zero backpropagation or fine-tuning.

| Threshold | Accuracy (%) | Recall (%) | Specificity (%) | Precision (%) | F1-score (%) | G-Mean (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.51 | 88.30 | 85.66 | 90.94 | 90.44 | 87.98 | 88.26 |
| **0.76** | **89.25** | **84.15** | **94.34** | **93.70** | **88.67** | **89.10** |
*Table 6.2: Performance (test metrics) of the proposed Late Fusion Model on external MIT-BIH AF dataset.*

### Comparison with State-of-the-Art (Cross-Dataset)
| Study | Model | Training Dataset | External Dataset | Precision (%) | Recall (%) | F1-score (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Menaceur et al. [17] | DSC-BiGRU | Chapman | MIT-BIH AF | 100.00 | 66.70 | 80.00 |
| Phukan et al. [34] | Wavelet + XGBoost | PhysioNet | CPSC2021 | - | - | 84.30 |
| **Proposed** | **Late Fusion** | **PhysioNet** | **MIT-BIH AF** | **93.70** | **84.15** | **88.67** |
*Table 6.3: Comparison of cross-dataset validation results with similar studies.*

## 🛠️ Installation & Setup
The entire pipeline was implemented in Python using the following dependencies:
*   `tensorflow` / `keras` 
*   `scipy` / `numpy`
*   `matplotlib` / `seaborn`
*   `pandas` (version 2.3.3)
*   `wfdb` (version 4.3.0/4.3.1)
*   `scikit-learn`
*   `PyWavelets` (`pywt`)

```bash
pip install tensorflow scipy numpy matplotlib seaborn scikit-learn PyWavelets
pip install pandas==2.3.3
pip install wfdb
