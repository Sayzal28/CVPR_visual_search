# Content-Based Image Retrieval (CBIR) System

This repository contains a Content-Based Image Retrieval (CBIR) system developed for the EEE3032 Computer Vision and Pattern Recognition module. The project implements and systematically evaluates several visual search techniques on the **MSRC-v2 dataset**.

The primary goal is to retrieve visually similar images for a given query by comparing feature descriptors. Performance is quantitatively measured using **Mean Average Precision (mAP)**.

Additionally, the most effective descriptors are validated in a secondary image classification task using a **Support Vector Machine (SVM)**.

## 🛠️ Techniques Implemented

### 1. Feature Descriptors
A range of descriptors were implemented to capture colour and texture information:

* **Global Colour Histogram (GCH):** The baseline method. It quantizes the 3D RGB space of the entire image into a single histogram, capturing the global colour profile while discarding spatial information.
* **Spatial Grid Descriptors:** To reintroduce spatial context, the image is divided into a uniform grid (e.g., 8x8). The following features are then extracted from each cell and concatenated:
    * **Average Colour:** A 3-dimensional (R, G, B) vector of the average colour in each cell.
    * **Local Colour Histogram:** A GCH applied individually to each grid cell.
    * **Edge Orientation Histogram (EOH):** A texture descriptor that uses a Sobel filter to find image gradients, then histograms the orientations of strong edges within each cell.
* **HSV Histogram:** A global histogram based on the Hue, Saturation, and Value (HSV) colour space, which is more aligned with human colour perception.
* **Local Binary Patterns (LBP):** A texture descriptor that works by comparing each pixel's brightness to its neighbours, creating a binary pattern to represent micro-textures.

### 2. Distance Metrics
The dissimilarity between feature descriptors was measured using:
* Euclidean Distance (L2 Norm)
* Manhattan Distance (L1 Norm)
* Chi-Squared Distance
* Cosine Distance
* Mahalanobis Distance (used with PCA)

### 3. Dimensionality Reduction & Classification
* **Principal Component Analysis (PCA):** Investigated as a technique to reduce the high dimensionality of combined feature vectors.
* **Support Vector Machine (SVM):** Used to evaluate the discriminative power of the best descriptors in a multi-class classification task (20 classes).

## 📊 Experimental Results

### CBIR (Retrieval) Performance
The following table summarizes the peak mAP achieved by the most notable descriptor and distance metric combinations.

| Method | Parameters | Distance Metric | Peak mAP |
| :--- | :--- | :--- | :--- |
| **Baseline: Global Colour Hist** | Q=16 | Chi-Squared | 0.2131 |
| EOH (Texture Only) | T=32 | L1 | 0.1527 |
| Spatial Grid (Avg. Colour + EOH) | T=16 | L1 | 0.1869 |
| **Optimal: Spatial Grid (Colour Hist + EOH)** | **Q=24, T=16** | **L1** | **0.2442** |
| HSV Histogram | H=64, S=32, V=32 | Chi-Squared | 0.2162 |
| LBP (Texture Only) | R=2 | Chi-Squared | 0.1748 |
| PCA (on best descriptor) | k=128 | Mahalanobis | 0.1727 |

### SVM (Classification) Performance
The best retrieval descriptor was compared against the baseline in a classification task using an 80:20 split.

| Descriptor | SVM Kernel | Classification Accuracy |
| :--- | :--- | :--- |
| Global Colour Histogram (Baseline) | Linear (C=0.01) | 39.50% |
| **Spatial Grid (Colour Hist + EOH)** | **Linear (C=0.01)** | **51.26%** |

## 💡 Key Findings & Conclusion

1.  **Feature Synergy is Critical:** The most significant finding was that combining local **colour** (Colour Histogram) and **texture** (EOH) information within a **spatial grid** yielded the best performance. The optimal mAP of 0.2442 was higher than any single-feature descriptor, demonstrating this combination is critical for discriminating between categories in the MSRC-v2 dataset.

2.  **Metric Choice Matters:** For high-dimensional, sparse histograms, the **L1 (Manhattan)** and **Chi-Squared** distance metrics consistently outperformed the **Euclidean (L2)** distance. This is because L2 is highly susceptible to the "Curse of Dimensionality," where L1 and Chi-Squared are more robust.

3.  **PCA was Ineffective:** Using PCA for dimensionality reduction was detrimental to performance. The best PCA-based score (mAP 0.1727) was significantly lower than the original, full-dimensional descriptor's score (mAP 0.2442). This suggests that PCA, while reducing size, discarded critical, discriminative information necessary for this task.

4.  **Retrieval Features Excel at Classification:** The SVM classification results reinforced the retrieval findings. The superior retrieval descriptor (Spatial Grid + Colour/EOH) achieved a **51.26%** accuracy, dramatically outperforming the baseline descriptor's **39.5%**, confirming its enhanced discriminative power.
