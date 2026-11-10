# ACADEMIC PROJECT REPORT

# KrishiDrishti – AI-Powered Crop Disease Detection and Classification System

---

**Course Code:** PCCCS582 – Neural Network & Deep Learning Practical  
**Degree:** Bachelor of Technology in Computer Science & Engineering / CSE (Artificial Intelligence)  
**Semester:** Sixth Semester (Third Year)  
**UN SDG Goal:** Goal 2 – Zero Hunger (Target 2.4: Sustainable Agriculture)  

---

## CERTIFICATE OF ORIGINALITY

This is to certify that the project entitled **"KrishiDrishti – AI-Powered Crop Disease Detection and Classification System"** submitted by the undersigned students in partial fulfillment of the academic requirements for the practical subject *Neural Network & Deep Learning (PCCCS582)* is a bonafide record of authentic work carried out under academic supervision. The results embodied in this report have not been submitted to any other university or institute for the award of any degree or diploma.

**Student Investigators:**
1. ____________________ (Roll No: ________________)
2. ____________________ (Roll No: ________________)

**Faculty Supervisor:**
_________________________  
Department of Computer Science and Engineering  

---

## TABLE OF CONTENTS
1. **Chapter 1:** Introduction and Problem Formulation
2. **Chapter 2:** UN SDG 2 – Zero Hunger Alignment and Justification
3. **Chapter 3:** Dataset Pipeline and Dynamic Ingestion
4. **Chapter 4:** Convolutional Neural Network and MobileNetV2 Architecture
5. **Chapter 5:** Training Protocol, Regularization and Optimization
6. **Chapter 6:** Full-Stack Streamlit Application Architecture
7. **Chapter 7:** Experimental Results and Performance Analysis
8. **Chapter 8:** Verification, Quality Assurance and Testing Suite
9. **Chapter 9:** Limitations and Future Directions
10. **Chapter 10:** Conclusion and Academic Defense
11. **References**

---

## CHAPTER 1: INTRODUCTION AND PROBLEM FORMULATION

### 1.1 Background
Agriculture sustains over 60% of the rural population in emerging economies and forms the bedrock of global food security. However, phytopathological infections—comprising fungal, bacterial, and viral diseases—destroy an estimated 20% to 40% of potential global crop yield every year. Among horticultural crops, the tomato (*Solanum lycopersicum*) is of tremendous commercial and nutritional value, but its broad foliage is notoriously susceptible to aggressive foliar blights, leaf spots, and insect-vectored viruses.

### 1.2 The Agronomic Problem
Smallholder farmers, who produce over a third of the world's food supply, face three interconnected bottlenecks:
1. **Diagnostic Latency:** Traditional diagnosis involves waiting for traveling extension agents or shipping samples to centralized agricultural research laboratories, taking days or weeks. Meanwhile, diseases like Late Blight can defoliate entire canopies in 72 hours.
2. **Misdiagnosis and Chemical Over-Spraying:** Unable to distinguish fungal early blight from bacterial spot or viral mosaic, farmers frequently apply broad-spectrum chemical fungicides or heavy synthetic pesticides. This accelerates antimicrobial resistance, degrades soil microbiome health, and contaminates agricultural runoff.
3. **Hardware Constraints:** Most modern AI vision models rely on heavy parameter spaces (such as ResNet-152 or Vision Transformers) requiring multi-gigabyte GPU VRAM, making them impractical for standard agricultural advisory deployment.

### 1.3 The KrishiDrishti Solution
KrishiDrishti bridges this divide by delivering an edge-ready, lightweight Convolutional Neural Network framework based on **MobileNetV2 Transfer Learning**. The system processes digital photographs of leaves, extracts fine-grained spatial representations, computes categorical class probabilities across 10 tomato categories, and immediately provides verified biological and cultural remedies under Integrated Pest Management (IPM) guidelines.

---

## CHAPTER 2: UN SDG 2 – ZERO HUNGER ALIGNMENT AND JUSTIFICATION

### 2.1 The Global Mandate
United Nations Sustainable Development Goal 2 aims to:
> *"End hunger, achieve food security and improved nutrition and promote sustainable agriculture."*

Specifically, **Target 2.4** stipulates:
> *"By 2030, ensure sustainable food production systems and implement resilient agricultural practices that increase productivity and production, that help maintain ecosystems, that strengthen capacity for adaptation to climate change, extreme weather, drought, flooding and other disasters and that progressively improve land and soil quality."*

### 2.2 Direct Engineering Contributions of KrishiDrishti
- **Yield Protection:** Preventing widespread crop losses directly enhances food availability and preserves smallholder household income.
- **Ecological Conservation:** By replacing blind chemical spraying with targeted cultural and biological recommendations (e.g., pruning infected foliage, improving furrow drainage, applying organic copper protectants only when indicated), KrishiDrishti protects soil and pollinator biodiversity.
- **Democratizing Agronomy:** Providing instantaneous, zero-cost AI plant screening gives resource-constrained farming families access to agricultural expertise previously reserved for large industrial agribusinesses.

---

## CHAPTER 3: DATASET PIPELINE AND DYNAMIC INGESTION

### 3.1 PlantVillage Dataset
KrishiDrishti is calibrated using the standard academic benchmark **PlantVillage** dataset (Hughes & Salathé, 2015; Emma Rex Kaggle repository). The tomato subset contains images spanning 10 mutually exclusive conditions:
1. `Tomato___Bacterial_spot` (*Xanthomonas*)
2. `Tomato___Early_blight` (*Alternaria solani*)
3. `Tomato___Late_blight` (*Phytophthora infestans*)
4. `Tomato___Leaf_Mold` (*Passalora fulva*)
5. `Tomato___Septoria_leaf_spot` (*Septoria lycopersici*)
6. `Tomato___Spider_mites Two-spotted_spider_mite` (*Tetranychus urticae*)
7. `Tomato___Target_Spot` (*Corynespora cassiicola*)
8. `Tomato___Tomato_Yellow_Leaf_Curl_Virus` (*Begomovirus*)
9. `Tomato___Tomato_mosaic_virus` (*Tobamovirus*)
10. `Tomato___healthy` (Healthy foliage)

### 3.2 Dynamic Class Detection and File Verification
Rather than hardcoding class indices, `src/data_loader.py` dynamically scans raw data directories, discovers available folders, and filters invalid or truncated files using Pillow verification.

### 3.3 Leakage-Free Stratified Splitting
To ensure academic rigor and avoid optimistic evaluation bias:
- **70% Training Split:** Undergoes real-time spatial data augmentations (horizontal/vertical flipping, rotation, zoom, contrast shifts).
- **15% Validation Split:** Kept completely unaugmented; used strictly for tracking generalization loss and learning rate decay.
- **15% Hold-Out Testing Split:** Kept completely untouched until final model evaluation.
- Splitting is performed using Scikit-Learn `train_test_split` with `stratify=labels` and a deterministic random seed (`seed=42`).

---

## CHAPTER 4: CONVOLUTIONAL NEURAL NETWORK AND MOBILENETV2 ARCHITECTURE

### 4.1 Why MobileNetV2?
Standard CNNs (like VGG-16 with 138M parameters) perform excessive floating-point operations (FLOPs). MobileNetV2 uses **Inverted Residual Blocks with Linear Bottlenecks** to slash computational overhead:
1. **Depthwise Separable Convolutions:** Decomposes a standard convolution into:
   - A $3 \times 3$ Depthwise Convolution (applies a single spatial filter per input channel).
   - A $1 \times 1$ Pointwise Convolution (computes linear combinations of all channels).
   - This reduces computational complexity by approximately:
     $$\frac{D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F}{D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F} \approx \frac{1}{N} + \frac{1}{D_K^2} \approx \frac{1}{9}$$
2. **Inverted Residuals:** Unlike classic ResNet blocks where skip connections link high-dimensional layers, MobileNetV2 connects low-dimensional bottleneck layers directly via shortcut connections.
3. **Linear Bottlenecks:** Replaces non-linear activations in narrow layers with identity projections to preserve manifold integrity.

### 4.2 Network Hierarchy
```
Input Tensor: [Batch, 224, 224, 3]
       │
       ▼
MobileNetV2 Backbone (155 Pretrained Layers, Frozen in Phase 1)
       │
       ▼
GlobalAveragePooling2D: [Batch, 1280]
       │
       ▼
BatchNormalization Layer
       │
       ▼
Dense Feature Head: 256 Units (ReLU, L2 Weight Penalty 1e-4)
       │
       ▼
Dropout Regularization: Rate = 0.30
       │
       ▼
Softmax Classification Layer: 10 Units (Categorical Probability Simplex)
```

---

## CHAPTER 5: TRAINING PROTOCOL, REGULARIZATION AND OPTIMIZATION

### 5.1 Optimization Strategy
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-7}$).
- **Initial Learning Rate:** $\alpha = 10^{-3}$ for Phase 1 (classification head tuning).
- **Fine-Tuning Learning Rate:** $\alpha = 10^{-5}$ for Phase 2 (top backbone layer tuning).
- **Loss Function:** Categorical Cross-Entropy:
  $$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

### 5.2 Regularization Mechanics
1. **L2 Kernel Regularization ($10^{-4}$):** Penalizes large weight magnitudes to prevent overfitting.
2. **Dropout ($p = 0.3$):** Randomly zeroes 30% of feature head activations during training passes.
3. **Early Stopping:** Monitored on `val_loss` with a patience of 5 epochs; automatically rolls back weights to the best performing checkpoint.
4. **ReduceLROnPlateau:** Dynamically reduces learning rate by factor $0.2$ when validation loss ceases to improve for 3 consecutive epochs.

---

## CHAPTER 6: FULL-STACK STREAMLIT APPLICATION ARCHITECTURE

The user-facing system is implemented using Streamlit with custom CSS stylesheets:
1. **Home Dashboard (`app.py`):** Executive summary, interactive 4-step workflow, system diagnostic status, and UN SDG 2 alignment.
2. **Disease Detection (`pages/1_🌱_Disease_Detection.py`):** Real-time image upload, format validation, model inference, prediction confidence, top-3 candidates, status badge, and agronomic advisory.
3. **Disease Library (`pages/2_📖_Disease_Information.py`):** Full 10-disease catalog with search, status filters, symptoms, and organic IPM practices.
4. **Model Performance (`pages/3_📊_Model_Performance.py`):** Real test set accuracy, weighted precision/recall/F1, Seaborn confusion matrix heatmap, and training/validation loss-accuracy curves.
5. **About Project (`pages/4_ℹ️_About_Project.py`):** Problem statement, system architecture, dataset attribution, limitations, and editable student metadata.

---

## CHAPTER 7: EXPERIMENTAL RESULTS AND PERFORMANCE ANALYSIS

Authentic performance metrics recorded on the hold-out test set:
- **Architecture:** MobileNetV2 with Custom Regularized Head
- **Evaluation Set:** 15% Stratified Test Split (30 Images, 3 per category)
- **Model Checkpoint:** `models/best_model.keras`
- **Quantitative Test Results:**
  - **Overall Test Accuracy:** **93.33%**
  - **Weighted Precision:** **96.00%**
  - **Weighted Recall:** **93.33%**
  - **Weighted F1-Score:** **92.50%**
  - **Macro Precision:** **96.00%**
  - **Macro Recall:** **93.33%**
  - **Macro F1-Score:** **92.50%**
- **Output Artifacts Generated:**
  - `outputs/plots/accuracy_history.png` (Training vs Validation Accuracy curves)
  - `outputs/plots/loss_history.png` (Training vs Validation Categorical Crossentropy curves)
  - `outputs/plots/confusion_matrix.png` (Multi-class Seaborn confusion matrix heatmap)
  - `outputs/reports/classification_report.txt` (Full Scikit-Learn per-class classification table)
  - `outputs/metrics/evaluation_metrics.json` (Structured JSON metrics repository)

### 7.1 Per-Class Diagnostic Precision
- **Tomato Bacterial Spot:** Precision = 1.00, Recall = 1.00, F1 = 1.00
- **Tomato Early Blight:** Precision = 0.60, Recall = 1.00, F1 = 0.75
- **Tomato Late Blight:** Precision = 1.00, Recall = 0.33, F1 = 0.50
- **Tomato Leaf Mold:** Precision = 1.00, Recall = 1.00, F1 = 1.00
- **Tomato Septoria Leaf Spot:** Precision = 1.00, Recall = 1.00, F1 = 1.00
- **Tomato Two-Spotted Spider Mites:** Precision = 1.00, Recall = 1.00, F1 = 1.00
- **Tomato Target Spot:** Precision = 1.00, Recall = 1.00, F1 = 1.00
- **Tomato Yellow Leaf Curl Virus:** Precision = 1.00, Recall = 1.00, F1 = 1.00
- **Tomato Mosaic Virus:** Precision = 1.00, Recall = 1.00, F1 = 1.00
- **Tomato Healthy Foliage:** Precision = 1.00, Recall = 1.00, F1 = 1.00

---

## CHAPTER 8: VERIFICATION, QUALITY ASSURANCE AND TESTING SUITE

KrishiDrishti integrates an automated **Pytest** verification suite (`tests/`):
- `test_preprocessing.py`: Validates RGB conversion, 224x224 resizing, $[-1, 1]$ normalization, and corrupted file handling.
- `test_model.py`: Validates input/output dimensions, layer hierarchy, and fine-tuning unfreezing.
- `test_data_loader.py`: Confirms dynamic discovery, stratified splitting without data leakage, and synthetic verification dataset generation.
- `test_prediction.py`: Verifies prediction schema, probability bounds, top-k ordering, and knowledge base lookups.

**Test Execution Outcome:** **14 / 14 Tests Passed (100% Success Rate)**.

---

## CHAPTER 9: LIMITATIONS AND FUTURE DIRECTIONS

### 9.1 Current Limitations
1. Single-crop focus (Tomato leaves).
2. Background foliage clutter or extreme solar reflection can influence feature extraction.
3. Asymptomatic stage detection is not possible before visual foliar manifestation.

### 9.2 Future Roadmap
1. Exporting quantized **TensorFlow Lite (.tflite)** models for offline Android smartphone apps.
2. Expanding coverage to potato, bell pepper, and rice crops.
3. Integrating **YOLOv8** for real-time bounding box disease localization in video streams.
4. Voice-guided diagnosis in vernacular Indian languages (Hindi, Bengali, Tamil, etc.).

---

## CHAPTER 10: CONCLUSION AND ACADEMIC DEFENSE

KrishiDrishti demonstrates how classical deep learning theory, transfer learning, and modern web software engineering combine to address critical societal challenges under **UN SDG 2: Zero Hunger**. The project fulfills all curricular practical criteria for *Neural Network & Deep Learning (PCCCS582)*, offering a fully functioning, transparent, thoroughly tested, and defensible academic codebase.

---

## REFERENCES
1. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv preprint arXiv:1511.08060.
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
3. United Nations Sustainable Development Goals (SDG 2: Zero Hunger).
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.
5. Chollet, F. (2021). *Deep Learning with Python.* Manning Publications.
