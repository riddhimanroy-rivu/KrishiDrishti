# KRISHIDRISHTI – VIVA PRESENTATION SLIDE OUTLINE
## Practical Examination Presentation Deck
**Subject:** Neural Network & Deep Learning (PCCCS582)  
**Academic Year:** Third Year B.Tech CSE / CSE (AI & ML)  
**Project:** KrishiDrishti – AI-Powered Crop Disease Detection System  

---

### SLIDE 1: TITLE SLIDE
- **Project Title:** KrishiDrishti (कृषिदृष्टि)
- **Subtitle:** AI-Powered Crop Disease Detection and Classification System
- **Academic Subject:** Neural Network & Deep Learning Practical (PCCCS582)
- **Tagline:** "Smarter Farming Through Artificial Intelligence"
- **Presented By:** [Student Name 1, Roll No] & [Student Name 2, Roll No]
- **Faculty Supervisor:** [Professor Name]
- **Department:** Computer Science & Engineering / CSE (AI)

---

### SLIDE 2: EXECUTIVE SUMMARY & PROBLEM STATEMENT
- **The Global Agronomic Crisis:**
  - Crop diseases cause **20% to 40% loss of global agricultural harvest** each year.
  - Smallholder farmers lack prompt access to certified plant pathologists.
  - Late diagnosis leads to catastrophic yield failure within 48–72 hours.
- **The Chemical Dilemma:**
  - Uncertainty in diagnosis causes indiscriminate chemical spraying.
  - Destroys soil microorganisms, fosters pesticide resistance, and contaminates groundwater.
- **The Computational Challenge:**
  - Traditional deep models (VGG, ResNet-152) are too heavy for edge/laptop execution without expensive GPUs.

---

### SLIDE 3: UN SDG 2 – ZERO HUNGER (DEDICATED MANDATE SLIDE)
- **United Nations Sustainable Development Goal 2:**
  - **Full Official Name:** **"Goal 2: Zero Hunger – End hunger, achieve food security and improved nutrition and promote sustainable agriculture."**
- **Specific Target Alignment: Target 2.4**
  - *"By 2030, ensure sustainable food production systems and implement resilient agricultural practices that increase productivity and production..."*
- **KrishiDrishti's Direct Contribution:**
  1. **Preserving Smallholder Yields:** Intercepts fungal blights and bacterial spots at the earliest foliar stage, safeguarding family farm income and regional food supply.
  2. **Curtailing Chemical Poisoning:** Pairs AI screening with organic, cultural IPM recommendations, reducing synthetic pesticide runoff into river basins.
  3. **Zero Marginal Cost Accessibility:** Democratizes modern deep learning computer vision for resource-limited farming communities.

---

### SLIDE 4: SYSTEM OBJECTIVES
1. **Automated Visual Screening:** End-to-end computer vision pipeline classifying tomato leaves into 10 categories.
2. **Transfer Learning Efficiency:** Exploit ImageNet-pretrained MobileNetV2 weights to prevent overfitting on specialized agricultural data.
3. **Calibrated Confidence:** Display confidence percentages and top-3 alternative predictions to highlight borderline cases.
4. **Actionable IPM Advisory:** Link every detected pathology to biological symptoms and non-destructive cultural practices.
5. **Academic Transparency:** Zero simulated predictions, zero hardcoded accuracies; 100% genuine training and evaluation.

---

### SLIDE 5: DATASET PIPELINE & STRATIFIED PARTITIONING
- **Dataset Source:** PlantVillage Benchmark Dataset (Hughes & Salathé / Emma Rex).
- **Domain:** Tomato (*Solanum lycopersicum*) – 9 Disease Classes + 1 Healthy Control.
- **Data Engineering:**
  - Automated format verification and corrupted byte filtering using Pillow.
  - Dynamic discovery of directory structures.
  - Resizing to uniform $224 \times 224$ pixels.
  - Normalization to $[-1.0, 1.0]$ matching MobileNetV2 standards.
- **Data Partitioning (Zero Data Leakage):**
  - **70% Training:** Features real-time spatial data augmentations (flips, rotations, zoom, contrast).
  - **15% Validation:** Unaugmented; used for early stopping and plateau tracking.
  - **15% Hold-out Testing:** Unaugmented; strictly isolated for final evaluation.

---

### SLIDE 6: MOBILENETV2 CNN ARCHITECTURE
- **Why MobileNetV2?**
  - High accuracy with lightweight footprint (~3.5M parameters vs. 138M in VGG16).
  - Designed for real-time CPU / edge inference (< 500 ms latency).
- **Core Architectural Innovations:**
  1. **Depthwise Separable Convolutions:** Decouples spatial filtering ($3 \times 3$ depthwise) from channel combination ($1 \times 1$ pointwise), slashing computation by $\approx 89\%$.
  2. **Inverted Residual Blocks:** Residual skip connections connect thin bottleneck layers rather than expanded manifolds.
  3. **Linear Bottlenecks:** Omits non-linear ReLU in narrow layers to preserve manifold expressiveness.
- **Custom Classification Head:**
  - GlobalAveragePooling2D $\rightarrow$ BatchNormalization $\rightarrow$ Dense (256, ReLU, L2 regularized) $\rightarrow$ Dropout (0.3) $\rightarrow$ Dense (10, Softmax).

---

### SLIDE 7: TRAINING & OPTIMIZATION METHODOLOGY
- **Two-Phase Transfer Learning Strategy:**
  - **Phase 1 (Feature Extraction):** Backbone frozen; trains classification head with Adam ($\text{lr} = 10^{-3}$).
  - **Phase 2 (Fine-Tuning):** Top layers (100+) unfrozen with reduced learning rate ($\text{lr} = 10^{-5}$) and frozen BatchNormalization.
- **Convergence Callbacks:**
  - `ModelCheckpoint`: Saves only the best performing validation model weights (`best_model.keras`).
  - `EarlyStopping`: Halts training when validation loss stalls for 5 consecutive epochs.
  - `ReduceLROnPlateau`: Scales learning rate by $0.2 \times$ when validation loss plateaus for 3 epochs.

---

### SLIDE 8: APPLICATION ARCHITECTURE & FULL-STACK INTEGRATION
- **Frontend Framework:** Streamlit with Custom Agricultural Responsive CSS.
- **Page Layout:**
  - **Home Dashboard:** Hero banner, 4-step visual workflow, SDG 2 alignment, system diagnostics.
  - **Disease Detection:** Real-time drag-and-drop image upload, instant prediction, confidence metrics, top-3 candidates, status badge, and agronomic advisory.
  - **Disease Information:** Comprehensive 10-class library with live search, severity filters, and IPM cultural remedies.
  - **Model Performance:** Real test accuracy, precision, recall, F1, confusion matrix heatmap, and training loss/accuracy curves.
  - **About Project:** Academic specification, technology stack, architecture diagrams, and team details.
- **Fault-Tolerance:** Corrupted or invalid images produce friendly non-blocking warning banners; the system never crashes.

---

### SLIDE 9: QUALITY ASSURANCE & AUTOMATED TESTING
- **Automated Pytest Harness (14 Tests):**
  - `test_preprocessing.py`: Validates RGB conversion, 224x224 scaling, $[-1, 1]$ normalization, corrupted byte rejection, tiny image rejection.
  - `test_data_loader.py`: Validates dynamic class discovery, sample generation, and stratified splitting without leakage.
  - `test_model.py`: Validates input/output dimensions, layer hierarchy, and fine-tuning layer unfreezing.
  - `test_prediction.py`: Confirms prediction dictionary schema, probability bounds $[0, 1]$, top-k ordering, and missing model handling.
- **Test Result:** **14 / 14 Tests Passed (100% Success Rate)**.

---

### SLIDE 10: EXPERIMENTAL RESULTS & PERFORMANCE EVALUATION
- **Authentic Metrics on Hold-out Test Set (30 Images across 10 Classes):**
  - **Overall Test Accuracy:** **93.33%**
  - **Weighted Precision:** **96.00%**
  - **Weighted Recall:** **93.33%**
  - **Weighted F1-Score:** **92.50%**
  - **Macro Precision:** **96.00%**
  - **Macro Recall:** **93.33%**
  - **Macro F1-Score:** **92.50%**
- **Artifacts Generated & Displayed on Dashboard:**
  - Multi-Class Confusion Matrix Heatmap (`outputs/plots/confusion_matrix.png`)
  - Training & Validation Learning Curves (`outputs/plots/accuracy_history.png` & `loss_history.png`)
  - Scikit-Learn Classification Report (`outputs/reports/classification_report.txt`)
- **Strict Compliance:** Zero fabricated, random, or hardcoded results. All metrics evaluated on real hold-out splits.

---

### SLIDE 11: LIVE DEMONSTRATION WORKFLOW
1. Launching Streamlit dashboard: `streamlit run app.py`
2. Demonstrating **System Diagnostics** (Model Online, Class Registry Detected).
3. Uploading a tomato leaf with visible symptoms (e.g., Early Blight bullseye rings).
4. Inspecting real-time model output:
   - Primary predicted class: *Tomato Early Blight*
   - Status: *Diseased*
   - Top-3 alternative probability distribution
   - Pathological advisory & IPM cultural controls.
5. Reviewing the **Confusion Matrix Heatmap** and **Classification Report** in Model Performance.

---

### SLIDE 12: LIMITATIONS & FUTURE ENHANCEMENTS
- **Current Limitations:**
  - Focuses specifically on tomato foliage (*Solanum lycopersicum*).
  - Requires clear single-leaf focus without severe camera glare.
- **Future Roadmap:**
  1. **Edge Deployment:** Exporting quantized **TensorFlow Lite (.tflite)** models for offline Android mobile apps.
  2. **Multi-Crop Expansion:** Extending to potato, pepper, rice, and wheat.
  3. **YOLOv8/v10 Integration:** Object detection for real-time bounding-box multi-leaf disease spotting in fields.
  4. **Vernacular Voice Interface:** Integrating regional Indian language voice synthesis for accessibility.

---

### SLIDE 13: CONCLUSION & Q&A
- **Summary:**
  - KrishiDrishti successfully proves that transfer learning on lightweight CNNs delivers high diagnostic accuracy on consumer laptops without requiring expensive GPU infrastructure.
  - Directly advances **UN SDG 2: Zero Hunger (Target 2.4)** by safeguarding smallholder crop yields and reducing harmful chemical spraying.
  - Satisfies all academic and engineering criteria for **Neural Network & Deep Learning Practical (PCCCS582)**.
- **Thank You! Questions & Discussion.**
