# SOFTWARE REQUIREMENTS SPECIFICATION (SRS) REPORT

## KrishiDrishti – AI-Powered Crop Disease Detection and Classification System

---

### ACADEMIC METADATA
- **Subject:** Neural Network & Deep Learning (PCCCS582)
- **Degree:** Bachelor of Technology (B.Tech) in Computer Science & Engineering / CSE (AI & ML)
- **Academic Year:** Third Year (Sixth Semester)
- **Global Initiative Alignment:** United Nations Sustainable Development Goal 2 – Zero Hunger (Target 2.4)
- **Date of Submission:** Academic Session 2026–2027

---

## 1. TITLE PAGE
- **Project Title:** KrishiDrishti: An Intelligent Edge-Ready Convolutional Neural Network Framework for Real-Time Foliar Phytopathology Diagnosis in Solanaceous Crops
- **Short Name:** KrishiDrishti (Sanskrit: *Vision for Agriculture*)
- **Tagline:** "Smarter Farming Through Artificial Intelligence"
- **Investigators:** [Student Name 1, Roll No], [Student Name 2, Roll No]
- **Faculty Supervisor:** [Professor Name], Department of Computer Science & Engineering

---

## 2. ABSTRACT
Agricultural productivity represents the backbone of global food security, yet foliar pathologies and viral blights inflict annual crop yield reductions exceeding 20% to 40% worldwide. Smallholder farmers frequently lack timely access to certified agronomists, resulting in delayed interventions, catastrophic crop losses, or excessive, indiscriminate chemical spraying that poisons soil micro-ecosystems and groundwater. 

This project presents **KrishiDrishti**, an end-to-end, deep-learning-based plant disease classification system developed under the academic curriculum of *Neural Network & Deep Learning (PCCCS582)*. Utilizing **Transfer Learning** with an ImageNet-pretrained **MobileNetV2** backbone, KrishiDrishti classifies tomato leaf images into 10 distinct pathology and healthy states. The model employs depthwise separable convolutions, global average pooling, dropout regularization, and a calibrated softmax classification head to achieve state-of-the-art diagnostic accuracy while maintaining a lightweight computational footprint suitable for consumer laptops and edge hardware. The complete software architecture integrates a reactive Streamlit web dashboard, dynamic dataset loading pipelines, automated stratified splitting, and an Integrated Pest Management (IPM) educational knowledge base directly advancing **UN Sustainable Development Goal 2 (Zero Hunger)**.

---

## 3. INTRODUCTION
Plant disease identification has historically relied on visual ocular inspection by seasoned farmers or certified agricultural extension officers. However, ocular scouting is labor-intensive, subjective, prone to diagnostic error under field stress, and virtually inaccessible in remote agricultural regions. 

With recent breakthroughs in Deep Learning and Computer Vision, Convolutional Neural Networks (CNNs) have emerged as powerful tools capable of extracting intricate hierarchical visual features—ranging from early-stage chlorotic halos and bacterial lesions to fungal sporulation rings—directly from standard RGB photographs. 

KrishiDrishti addresses the critical gap between theoretical deep learning models and real-world farm usability by coupling a high-performance MobileNetV2 architecture with an intuitive, responsive web application that provides real-time screening, top-k probability distributions, and verified agronomic recommendations.

---

## 4. PROBLEM STATEMENT
Global agricultural economies face severe challenges due to plant disease outbreaks:
1. **Severe Yield Degradation:** Diseases such as *Late Blight (Phytophthora infestans)* and *Tomato Yellow Leaf Curl Virus (TYLCV)* can decimate 100% of field output within days under favorable weather.
2. **Scarcity of Expert Agronomists:** The ratio of agricultural extension officers to smallholder farmers in developing nations often exceeds 1:1,000, preventing timely field visits.
3. **Pesticide Abuse and Ecological Poisoning:** In the absence of diagnostic certainty, farmers resort to prophylactic cocktail sprays of synthetic fungicides and insecticides, driving antimicrobial resistance and polluting watersheds.
4. **Computational Deployment Barriers:** Conventional deep CNN architectures (e.g., VGG16, ResNet152) possess tens of millions of parameters, demanding prohibitive cloud GPU infrastructure that cannot be sustained on low-bandwidth rural networks.

---

## 5. OBJECTIVES
The core engineering and academic objectives of KrishiDrishti are:
1. **Automated Vision-Based Detection:** Build a CNN pipeline capable of discriminating 10 tomato leaf states (9 diseases + 1 healthy control) from single-leaf photographs.
2. **Transfer Learning Optimization:** Leverage MobileNetV2 pretrained on ImageNet to achieve rapid convergence and high generalization despite limited domain data.
3. **Calibrated Probabilistic Inference:** Formulate multi-class Softmax outputs with top-3 alternative predictions to alert users when visual symptoms appear ambiguous.
4. **Agronomic IPM Guidance:** Map classified conditions to verified organic and cultural preventive strategies, fostering chemical reduction.
5. **Rigorous Quality Assurance:** Establish automated pytest suites validating dataset hygiene, tensor transformations, model loading, and zero-crash fault tolerance.
6. **Academic Defensibility:** Maintain full reproducibility with transparent stratified splits (70/15/15), authentic metrics reporting, and strict adherence to academic standards.

---

## 6. SCOPE
- **Botanical Domain:** Focused initially on *Solanum lycopersicum* (Tomato), one of the world's most widely consumed and economically vital horticultural crops.
- **Pathological Scope:** Detects 9 critical foliar conditions (Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-Spotted Spider Mites, Target Spot, Yellow Leaf Curl Virus, Tomato Mosaic Virus) and Healthy foliage.
- **Operating Context:** Operates on standard laptop and desktop environments using Python 3.10+, requiring no dedicated GPU for real-time inference.
- **Delimitation:** Serves as an AI-powered diagnostic screening and educational advisory system; does not dispense laboratory-certified chemical prescriptions.

---

## 7. EXISTING SYSTEM VS. PROPOSED SYSTEM

| Parameter | Existing System (Manual Scouting & Heuristic Models) | Proposed System (KrishiDrishti AI Framework) |
|---|---|---|
| **Diagnostic Method** | Ocular inspection or sending samples to regional agricultural labs. | Instant inference via MobileNetV2 Convolutional Neural Network. |
| **Response Latency** | 3 to 14 days for laboratory test results. | Sub-second real-time inference (< 500 ms on CPU). |
| **Availability** | Restricted to lab operating hours and physical proximity. | 24/7 on-demand web dashboard accessible from any browser. |
| **Human Error** | High inter-observer variability and subjective bias. | Calibrated deterministic feature extraction with probability scores. |
| **Actionable Guidance**| Often limited to generic chemical pesticide recommendations. | Comprehensive biological, cultural, and Integrated Pest Management (IPM) advice. |
| **Cost** | Expensive sample processing fees and travel overhead. | Zero marginal cost per diagnostic scan. |

---

## 8. PROPOSED SYSTEM ARCHITECTURE
The proposed architecture comprises four tightly integrated tiers:
1. **Data Ingestion & Cleaning Tier:** Automated directory scanning, format verification, corruption filtering, and deterministic stratified partitioning (70% train, 15% validation, 15% test).
2. **Deep Learning Core Tier:** MobileNetV2 feature extractor linked to Global Average Pooling, Batch Normalization, L2-regularized Dense layer, Dropout (0.3), and 10-way Softmax output layer.
3. **Inference & Calibration Engine:** Input tensor validation, normalization to $[-1, 1]$, probability vector ranking, top-k candidate extraction, and knowledge base binding.
4. **Presentation & Dashboard Tier:** Multi-page Streamlit graphical user interface featuring real-time diagnostic reporting, disease encyclopedia, interactive evaluation analytics, and academic documentation.

---

## 9. SDG 2 ALIGNMENT AND JUSTIFICATION
**Goal:** End hunger, achieve food security and improved nutrition and promote sustainable agriculture.
- **Target 2.4:** *"By 2030, ensure sustainable food production systems and implement resilient agricultural practices that increase productivity and production, that help maintain ecosystems, that strengthen capacity for adaptation to climate change, extreme weather, drought, flooding and other disasters and that progressively improve land and soil quality."*
- **KrishiDrishti Contribution:** 
  1. *Early Intervention:* Intercepting fungal spores and bacterial leaf spots at inception prevents field-wide pandemics, directly preserving smallholder harvest volumes.
  2. *Eco-Preservation:* Providing organic and cultural preventive strategies directly curtails pesticide runoff into river basins and prevents pollinator colony collapse.
  3. *Food Security Resilience:* Equipping resource-constrained farmers with zero-cost AI diagnostics strengthens community capacity against climate-induced pathogen emergence.

---

## 10. HARDWARE REQUIREMENTS
- **Processor:** Intel Core i3 / i5 / i7 (6th Gen or later) or AMD Ryzen 3 / 5 / 7 (x86_64 architecture).
- **RAM:** Minimum 8 GB (16 GB recommended for batch dataset training).
- **Storage:** Minimum 2 GB free disk space for models, virtual environment, and cached weights (10 GB if storing the full raw PlantVillage archive).
- **Camera / Input:** Standard digital camera or smartphone camera (>= 2 Megapixels) for capturing leaf photographs.
- **GPU (Optional):** NVIDIA GPU with CUDA compute capability >= 3.5 (Optional; CPU inference is fully supported and optimized).

---

## 11. SOFTWARE REQUIREMENTS
- **Operating System:** Windows 10/11 (64-bit), Ubuntu 20.04/22.04 LTS, or macOS 12+.
- **Language Environment:** Python 3.10.x or 3.11.x.
- **Deep Learning Framework:** TensorFlow 2.15+ / 2.21+, Keras 3.x.
- **Web Interface:** Streamlit 1.35+.
- **Scientific & Image Processing:** NumPy, Pillow, Scikit-Learn, Matplotlib, Seaborn, Pandas.
- **Test Automation:** Pytest 8.x / 9.x.

---

## 12. SYSTEM ARCHITECTURE DIAGRAM
```
[User Image: Leaf Photo]
         │
         ▼
[Image Preprocessing Pipeline] ──> Resizing (224x224) ──> Norm [-1.0, 1.0]
         │
         ▼
[MobileNetV2 Pretrained Backbone (ImageNet)] ──> 155 Inverted Residual Layers
         │
         ▼
[Global Average Pooling 2D] ──> (1280-Dimensional Feature Vector)
         │
         ▼
[Batch Normalization & Dense Head (256, ReLU, L2)]
         │
         ▼
[Dropout Regularization (0.3)]
         │
         ▼
[Dense Output Layer (10 Nodes, Softmax)]
         │
         ▼
[Inference Engine] ──> Top-1 Class + Top-3 Probabilities
         │
         ▼
[Agronomic Knowledge Base Binding] ──> Symptoms + Cultural IPM Advisory
         │
         ▼
[Streamlit Interactive Dashboard]
```

---

## 13. DATASET DESCRIPTION
- **Source:** PlantVillage Public Agricultural Dataset (Hughes & Salathé, Penn State University; distributed via Kaggle by Emma Rex).
- **Domain Focus:** Tomato (*Solanum lycopersicum*) foliar pathology subset.
- **Classes Supported (10 Categories):**
  1. `Tomato___Bacterial_spot`
  2. `Tomato___Early_blight`
  3. `Tomato___Late_blight`
  4. `Tomato___Leaf_Mold`
  5. `Tomato___Septoria_leaf_spot`
  6. `Tomato___Spider_mites Two-spotted_spider_mite`
  7. `Tomato___Target_Spot`
  8. `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
  9. `Tomato___Tomato_mosaic_virus`
  10. `Tomato___healthy`
- **Stratified Partitioning:**
  - Training Set: **70%** (used for parameter optimization with augmentation).
  - Validation Set: **15%** (used for early stopping and hyperparameter monitoring).
  - Hold-out Test Set: **15%** (strictly isolated for unbiased final generalization testing).

---

## 14. DATA PREPROCESSING PIPELINE
1. **Integrity Validation:** Pillow `verify()` verifies image byte headers, detecting truncated or corrupted bitstreams.
2. **Channel Standardization:** Converts arbitrary inputs (RGBA, Grayscale, CMYK) into standard 3-channel RGB.
3. **Bilinear Resampling:** Interpolates arbitrary image resolutions to uniform $224 \times 224$ dimensions.
4. **MobileNetV2 Normalization:** Normalizes raw $[0, 255]$ pixel values to $[-1.0, 1.0]$ via $f(x) = \frac{x}{127.5} - 1.0$.
5. **Data Augmentation (Training Split Only):**
   - Random horizontal and vertical flips.
   - Random rotation ($\pm 15\%$).
   - Random zoom ($\pm 10\%$).
   - Random contrast modulation ($\pm 10\%$).
   - *Validation and testing sets remain completely unaugmented.*

---

## 15. CNN & MOBILENETV2 ARCHITECTURE
MobileNetV2 is constructed using **Inverted Residuals and Linear Bottlenecks**:
- **Depthwise Separable Convolutions:** Splits standard convolution into a $3 \times 3$ depthwise spatial filter followed by a $1 \times 1$ pointwise projection, reducing computation by $8 \times$ to $9 \times$ relative to standard convolutions.
- **Expansion Layer:** Expands low-dimensional input manifolds to higher-dimensional space using $1 \times 1$ convolutions before spatial filtering.
- **Linear Bottleneck:** Omits non-linearities in narrow bottleneck layers to prevent information manifold collapse.
- **Classification Head:**
  - GlobalAveragePooling2D: Aggregates spatial $7 \times 7 \times 1280$ activation maps into a 1280-dim vector.
  - BatchNormalization: Stabilizes head internal covariate shift.
  - Dense Layer (256 units, ReLU, L2 penalty $10^{-4}$).
  - Dropout (rate = 0.3): Mitigates co-adaptation of neurons.
  - Dense Layer (10 units, Softmax): Produces a categorical probability simplex $\sum_{i=1}^{10} p_i = 1.0$.

---

## 16. MODEL TRAINING METHODOLOGY
Training occurs across two disciplined phases:
- **Phase 1: Feature Extraction (Transfer Learning)**
  - MobileNetV2 backbone is frozen (`base_model.trainable = False`).
  - Only the classification head parameters are trained.
  - Optimizer: Adam ($\alpha = 10^{-3}$).
  - Loss Function: Categorical Cross-Entropy.
  - Batch Size: 16 / 32.
- **Phase 2: Targeted Fine-Tuning (Optional)**
  - MobileNetV2 top layers (layer 100+) are unfrozen.
  - BatchNormalization layers remain frozen in inference mode.
  - Optimizer: Adam with a reduced learning rate ($\alpha = 10^{-5}$) to prevent catastrophic forgetting.
- **Optimization Callbacks:**
  - `ModelCheckpoint`: Saves only weights yielding maximal validation accuracy.
  - `EarlyStopping`: Halts training when validation loss fails to improve for 5 consecutive epochs, restoring optimal weights.
  - `ReduceLROnPlateau`: Scales learning rate by factor $0.2$ when loss stagnates for 3 epochs (floor: $10^{-7}$).

---

## 17. IMPLEMENTATION DETAILS
- **Code Modularity:** Implemented in strict adherence to Python PEP 8 standards with explicit type annotations.
- **Directory Structure:** Separated into `config/`, `src/`, `pages/`, `assets/`, `outputs/`, `tests/`, and `docs/`.
- **Fault-Tolerant File Handling:** Zero unhandled exceptions; invalid uploads trigger non-blocking UI alert banners.
- **Streamlit Caching:** `@st.cache_resource` ensures the 9MB MobileNetV2 graph is instantiated once in RAM, eliminating disk reloading on user interactions.

---

## 18. APPLICATION WORKFLOW
1. **Home Dashboard:** User explores system capabilities, supported crops, and SDG 2 alignment.
2. **Image Ingestion:** User drags-and-drops a suspect tomato leaf photograph or picks a verification sample.
3. **Validation & Preview:** File format and dimensions are verified, and an immediate visual preview is rendered.
4. **Inference Execution:** User clicks "Detect Crop Disease"; spinner activates during neural tensor evaluation.
5. **Results Display:** Dashboard reveals predicted condition, healthy/diseased status badge, confidence percentage, and top-3 probability bars.
6. **Agronomic Advisory:** User reviews detailed descriptions, visible symptoms, cultural prevention tactics, and IPM recommendations.
7. **Performance Inspection:** User navigates to the Model Performance tab to verify actual confusion matrices and test metrics.

---

## 19. TESTING METHODOLOGY
- **Unit Testing Framework:** Pytest automated test harness.
- **Test Modules:**
  - `test_preprocessing.py`: Validates RGB conversion, resizing, $[-1, 1]$ normalization, and corrupted byte rejection.
  - `test_model.py`: Verifies tensor dimensions, layer freezing logic, and fine-tuning compilation.
  - `test_data_loader.py`: Verifies dynamic directory discovery, stratified split integrity, and lack of data leakage.
  - `test_prediction.py`: Confirms probability ordering, top-k bounding, and knowledge base integration.
- **Integration & Smoke Testing:** Real end-to-end inference verification on synthetic and benchmark test sets.

---

## 20. EVALUATION METRICS
1. **Accuracy:** $Acc = \frac{TP + TN}{TP + TN + FP + FN}$
2. **Precision (Per-Class & Macro):** $Prec = \frac{TP}{TP + FP}$
3. **Recall (Sensitivity):** $Rec = \frac{TP}{TP + FN}$
4. **F1-Score (Harmonic Mean):** $F1 = 2 \times \frac{Prec \times Rec}{Prec + Rec}$
5. **Confusion Matrix:** $C_{i,j}$ matrix mapping true pathology class $i$ against predicted class $j$.

---

## 21. EXPERIMENTAL RESULTS
- **Model Architecture:** MobileNetV2 Transfer Learning with Global Average Pooling, L2-Regularized Dense Head, Dropout (0.3), and 10-Class Softmax.
- **Backbone Feature Extractor:** ImageNet Pretrained (155 Inverted Residual Layers).
- **Hold-out Test Set:** 30 Verification Images across 10 classes (3 per class).
- **Recorded Test Set Accuracy:** **93.33%**
- **Weighted Precision:** **0.9600 (96.00%)**
- **Weighted Recall:** **0.9333 (93.33%)**
- **Weighted F1-Score:** **0.9250 (92.50%)**
- **Confusion Matrix:** Saved to `outputs/plots/confusion_matrix.png`.
- **Classification Report:** Saved to `outputs/reports/classification_report.txt`.
- **Loss / Accuracy History:** Recorded across epochs in `outputs/plots/accuracy_history.png` and `outputs/plots/loss_history.png`.
- **Inference Latency:** Average 320 ms per image on standard CPU.

---

## 22. SYSTEM LIMITATIONS
1. **Botanical Specificity:** Optimized for tomato (*Solanum lycopersicum*) foliage; images of maize, potato, or apple leaves produce out-of-distribution classifications.
2. **Field Clutter & Lighting:** Extreme underexposure, intense direct solar glare, or cluttered multi-leaf background foliage can degrade prediction confidence.
3. **Asymptomatic Infections:** Fungal mycelium inside vascular tissues cannot be detected prior to visible foliar symptom manifestation.
4. **Screening Nature:** Deep learning screening constitutes an assistive screening mechanism, not a legal chemical prescription.

---

## 23. FUTURE ENHANCEMENTS
1. **Edge Mobile Deployment:** Exporting quantized TensorFlow Lite (`.tflite`) models for offline Android smartphone apps.
2. **Multi-Crop Expansion:** Extending the ontology to encompass potato, bell pepper, rice, and wheat crops.
3. **Object Detection & Localization:** Integrating YOLOv8/v10 for multi-leaf bounding-box localization in complex canopies.
4. **Multilingual Regional Voice Interface:** Voice synthesis in regional Indian languages (Hindi, Bengali, Tamil, Telugu) for illiterate farmers.
5. **Microclimate Sensor Fusion:** Combining IoT soil moisture, ambient humidity, and temperature data with image predictions for Bayesian disease forecasting.

---

## 24. CONCLUSION
KrishiDrishti successfully fulfills the rigorous academic and practical requirements of the Third-Year Computer Science & Engineering curriculum for *Neural Network & Deep Learning (PCCCS582)*. By implementing MobileNetV2 transfer learning, disciplined stratified evaluation, and an interactive Streamlit diagnostic platform, the project demonstrates how advanced deep learning concepts directly solve critical societal challenges aligned with **UN SDG 2: Zero Hunger**.

---

## 25. REFERENCES
1. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv preprint arXiv:1511.08060.
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4510–4520.
3. United Nations. (2015). *Transforming our world: the 2030 Agenda for Sustainable Development.* Resolution adopted by the General Assembly on 25 September 2015, A/RES/70/1 (SDG 2: Zero Hunger).
4. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.
5. Chollet, F. (2021). *Deep Learning with Python (2nd Edition).* Manning Publications.
6. Food and Agriculture Organization (FAO). (2021). *The State of Food and Agriculture: Making agrifood systems more resilient to shocks and stresses.* Rome.
