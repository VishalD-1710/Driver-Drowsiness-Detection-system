<img width="1200" height="600" alt="fatigue_progression_curve" src="https://github.com/user-attachments/assets/810d622b-8ef2-448e-968f-4b52f3a5ae1d" />
<img width="750" height="600" alt="confusion_matrix_MobileNetV2" src="https://github.com/user-attachments/assets/a9a2ae30-faff-4686-9a47-41aaf76c4d33" />
<img width="750" height="600" alt="confusion_matrix_CustomCNN" src="https://github.com/user-attachments/assets/1fa3a929-bf50-4e36-b9fb-08346f810568" />
<img width="1950" height="675" alt="accuracy_loss_comparison" src="https://github.com/user-attachments/assets/7019fbb5-116c-4b54-973b-1f64f6358bfe" />
# Driver-Drowsiness-Detection-system
Markdown# Driver Drowsiness Detection System Using Eye Closure & Yawning Analysis
An end-to-end vision-based driver monitoring pipeline designed to detect physiological signs of fatigue (eye closure and yawning) and map them into actionable 3-stage driver fatigue levels.[Driver_Drowsiness_Detection.ipynb](https://github.com/user-attachments/files/30425114/Driver_Drowsiness_Detection.ipynb)
[Driver_Drowsiness_Detection_Report.pdf](https://github.com/user-attachments/files/30425110/Driver_Drowsiness_Detection_Report.pdf)

---


---

## 👁️ Overview
Driver fatigue significantly impairs reaction time and decision-making, accounting for a high percentage of road accidents globally. Traditional methods relying on steering behavior or intrusive wearable sensors are often unreliable or uncomfortable. 

This project implements a non-intrusive, deep-learning-powered solution that continuously evaluates driver face imagery across four primary states: **Eyes Open**, **Eyes Closed**, **Yawn**, and **No Yawn**. It converts raw visual predictions into a **3-stage fatigue progression framework** to trigger timely safety interventions.

---

## ⚙️ Key Features
* **Dual Feature Analysis:** Monitors both ocular (eye closure) and oral (yawning) physiological indicators.
* **Transfer Learning:** Employs pre-trained MobileNetV2 for light-weight, high-accuracy feature extraction.
* **Robust Augmentation:** Handles real-world lighting variations, rotations, and subtle head pose shifts.
* **3-Level Decision Fusion:** Integrates 4-class multi-label output into distinct safety states (`Alert`, `Mild Fatigue`, `Severe Fatigue`).
* **Time-Series Progression Tracking:** Plots driver alert state deterioration across driving intervals to identify transition thresholds.

---

## 📁 Dataset Architecture
The raw dataset consists of standardized facial images partitioned across four core categories:

```text
dataset/
├── Closed/    # Eyes closed states
├── Open/      # Eyes open states
├── no_yawn/   # Mouth closed / normal facial expressions
└── yawn/      # Mouth open / yawning expressions
Image Dimensions: $224 \times 224 \times 3$Normalization: Pixel values scaled to $[0, 1]$ range.Augmentation Techniques: Random rotation ($\pm 15^\circ$), horizontal flipping, zoom ($\pm 15\%$), and brightness shift.🏗️ Model Architecture & MethodologyPreprocessing Layer: Input scaling and normalization.Feature Extraction (MobileNetV2 Base): Pre-trained on ImageNet with frozen base layers to extract low-level facial feature maps efficiently.Classification Head:Global Average Pooling 2DDense Layer (128 units, ReLU activation)Dropout (30% probability for regularization)Softmax Output Layer (4 units: Closed, Open, no_yawn, yawn)🔀 Decision Fusion LogicRather than presenting raw multi-class probabilities, the output predictions are fused using a rule-based state mapper to assign immediate threat levels:4-Class PredictionStage CodeFatigue LevelRecommended System ActionOpen / no_yawn0AlertNormal Operationyawn1Mild FatigueVisual Warning / Soft ChimeClosed2Severe FatigueLoud Alarm / Haptic Steering Vibration📈 Fatigue Progression AnalysisThe pipeline aggregates multi-frame sequences to model continuous driving sessions. By converting sequential frame predictions into time-based interval averages, the system outputs a Driver Fatigue Progression Curve, identifying critical transitions ($Alert \rightarrow Mild \rightarrow Severe$) over time.🚀 Installation & UsagePrerequisitesPython 3.8+TensorFlow 2.xOpenCV, NumPy, Matplotlib, Seaborn, Scikit-LearnSetupBash# Clone the repository
git clone [https://github.com/your-username/driver-drowsiness-detection.git](https://github.com/your-username/driver-drowsiness-detection.git)
cd driver-drowsiness-detection

# Install dependencies
pip install -r requirements.txt
Running the NotebookOpen Driver_Drowsiness_Detection.ipynb in Google Colab or Jupyter Notebook and execute all cells sequentially.📊 Results & PerformanceValidation Accuracy: $> 95\%$Evaluation Metrics: Evaluated using Precision, Recall, F1-Score, and Multi-class Confusion Matrices.Inference Speed: Lightweight MobileNetV2 architecture allows real-time execution suitable for edge deployment in Advanced Driver Assistance Systems (ADAS).

