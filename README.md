# 🛡️ Credit Card Fraud Detection

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

*A machine learning project leveraging deep learning to detect fraudulent credit card transactions with high accuracy*

[📊 View Dataset](#-dataset) • [🚀 Quick Start](#-quick-start) • [📖 Documentation](#-documentation)

</div>

---

## 🎯 Project Overview

This project develops a **sophisticated neural network model** to identify fraudulent credit card transactions from a real-world dataset. Despite the extreme class imbalance (only 0.17% fraudulent cases), the model achieves robust fraud detection through advanced deep learning techniques.

### 🔑 Key Objectives

- ✅ Build a high-precision fraud detection system
- ✅ Handle severe class imbalance effectively
- ✅ Implement robust data preprocessing and feature scaling
- ✅ Create interpretable visualizations for data insights
- ✅ Deploy production-ready machine learning model

---

## 📊 Dataset

<table align="center">
<tr>
<td>

### 📈 Quick Stats

| Metric | Value |
|--------|-------|
| **Total Transactions** | 284,807 |
| **Features** | 31 |
| **Legitimate (0)** | 99.83% |
| **Fraudulent (1)** | 0.17% |
| **Duplicates** | 1,081 |
| **Missing Values** | 0 |
| **Amount Range** | €0 - €25,691.16 |
| **Time Range** | 0 - 172,792 seconds |

</td>
<td>

### 🏦 Dataset Source

- **Platform**: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Organization**: Machine Learning Group - ULB
- **Region**: European cardholders
- **Time Period**: September 2013
- **Access**: Via kagglehub API

</td>
</tr>
</table>

### 📝 Feature Description

| Feature | Description |
|---------|-------------|
| **Time** | Seconds elapsed between transaction and first transaction |
| **V1 - V28** | PCA-transformed features (for privacy protection) |
| **Amount** | Transaction amount in EUR |
| **Class** | Target variable: 0 (Legitimate) / 1 (Fraudulent) |

---

## 🛠️ Technology Stack

<div align="center">

```
┌─────────────────────────────────────────┐
│      🐍 Python & Data Science           │
├─────────────────────────────────────────┤
│ 📦 pandas       - Data manipulation     │
│ 🔢 NumPy        - Numerical computing   │
│ 📊 Seaborn      - Statistical plots     │
│ 📈 Matplotlib   - Data visualization    │
│ 🎨 Plotly       - Interactive charts    │
├─────────────────────────────────────────┤
│      🤖 Machine Learning & Deep Learning │
├─────────────────────────────────────────┤
│ 🧠 TensorFlow/Keras - Neural networks   │
│ 📉 scikit-learn     - ML algorithms     │
│ ⚙️ kagglehub        - Dataset access    │
└─────────────────────────────────────────┘
```

</div>

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone the repository
git clone https://github.com/MrMaher19/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Setup Kaggle Credentials (for local use)

```bash
# Place your kaggle.json in ~/.kaggle/
# Download from: https://www.kaggle.com/settings/account
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3️⃣ Run the Project

```bash
# Open Jupyter notebook
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

### 4️⃣ Explore Results

- 📊 View EDA visualizations
- 📈 Analyze class distribution
- 🧠 Train the neural network
- 📉 Evaluate model performance

---

## 📖 Documentation

### 📋 Workflow Overview

```mermaid
graph LR
    A["📥 Data Loading"] --> B["🔍 EDA & Analysis"]
    B --> C["📊 Visualization"]
    C --> D["🧹 Data Preprocessing"]
    D --> E["⚖️ Handle Imbalance"]
    E --> F["🧠 Model Training"]
    F --> G["📈 Model Evaluation"]
    G --> H["✅ Deployment Ready"]
    
    style A fill:#e1f5ff
    style B fill:#f3e5f5
    style C fill:#e8f5e9
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#f1f8e9
    style G fill:#e0f2f1
    style H fill:#c8e6c9
```

### 🔄 Project Pipeline

#### **Phase 1: Data Exploration** 📊
```python
✓ Load dataset using kagglehub API
✓ Statistical summary analysis
✓ Check for missing values
✓ Identify duplicates
✓ Analyze class distribution
```

#### **Phase 2: Visualization** 📈
```python
✓ Count plot - Class distribution
✓ Histograms - Amount & Time distribution
✓ Box plots - Outlier detection
✓ Scatter plots - Fraud pattern analysis
✓ Interactive Plotly charts
```

#### **Phase 3: Preprocessing** 🧹
```python
✓ Feature scaling with RobustScaler
✓ Handle outliers
✓ Address class imbalance
✓ Train-test split
✓ Feature normalization
```

#### **Phase 4: Model Development** 🧠
```python
✓ Build Sequential Neural Network
✓ Implement Dropout (0.5)
✓ Add Batch Normalization
✓ Configure EarlyStopping
✓ Optimize hyperparameters
```

#### **Phase 5: Evaluation** 📊
```python
✓ Calculate accuracy metrics
✓ Generate confusion matrix
✓ Plot ROC-AUC curve
✓ Analyze precision & recall
✓ Model interpretation
```

---

## 🧠 Model Architecture

<div align="center">

### Neural Network Structure

```
Input Layer (30 features)
        ↓
Dense Layer (128) + ReLU + BatchNorm
        ↓
Dropout (0.5)
        ↓
Dense Layer (64) + ReLU + BatchNorm
        ↓
Dropout (0.5)
        ↓
Dense Layer (32) + ReLU + BatchNorm
        ↓
Dropout (0.5)
        ↓
Output Layer (1) + Sigmoid
        ↓
Binary Classification (Fraud / Legitimate)
```

### ⚙️ Model Configuration

| Parameter | Value |
|-----------|-------|
| **Loss Function** | Binary Crossentropy |
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Activation (Hidden)** | ReLU |
| **Activation (Output)** | Sigmoid |
| **Regularization** | Dropout (0.5) |
| **Batch Normalization** | ✅ Enabled |
| **Early Stopping** | ✅ Enabled |

</div>

---

## 📊 Key Insights & Findings

### 🔍 Data Quality

| Finding | Details |
|---------|---------|
| ✅ **No Missing Values** | Dataset is complete with 100% coverage |
| 🔄 **1,081 Duplicates** | Removed before final analysis |
| 📐 **Already Scaled** | V1-V28 are PCA-transformed features |
| ⚠️ **Severe Imbalance** | Only 0.17% fraud cases |

### 💡 Distribution Patterns

- **Amount**: Highly skewed with long right tail
- **Time**: Relatively uniform across 2-day period
- **Frauds**: Higher amounts but broader variance
- **Legitimate**: Concentrated around lower amounts

### 🎯 Class Imbalance Challenge

```
Non-Fraud: ████████████████████████████ 99.83%
Fraud:     ▌ 0.17%
```

**Impact**: Standard accuracy metrics misleading
**Solution**: Use precision, recall, F1-score, AUC-ROC

---

## ⚖️ Handling Class Imbalance

### Implemented Strategies

| Strategy | Description | Impact |
|----------|-------------|--------|
| **Class Weights** | Penalize misclassification of minority class | ⬆️ Model sensitivity |
| **Undersampling** | Reduce majority class samples | ⬇️ Data size |
| **Oversampling** | Increase minority class samples | ⬆️ Training time |
| **SMOTE** | Synthetic minority oversampling | 📈 Better generalization |
| **Threshold Tuning** | Adjust decision boundary | ⚖️ Precision-Recall trade-off |

---

## 📈 Performance Metrics

<table>
<tr>
<td>

### 🎯 Primary Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| **Precision** | Of predicted fraud, how many correct? | 85%+ |
| **Recall** | Of actual fraud, how many caught? | 80%+ |
| **F1-Score** | Harmonic mean of precision & recall | 0.82+ |
| **AUC-ROC** | Overall discrimination ability | 0.90+ |

</td>
<td>

### 📊 Confusion Matrix

```
                 Predicted
              Fraud | Legit
Actual  Fraud   TP   | FN
        Legit   FP   | TN
```

- **TP**: Correctly identified frauds
- **FN**: Missed frauds (costly)
- **FP**: False positives
- **TN**: Correctly identified legitimate

</td>
</tr>
</table>

---

## 📁 Project Structure

```
Credit_Card_Fraud_Detection/
│
├── 📓 Credit_Card_Fraud_Detection.ipynb  (Main notebook)
├── 📄 README.md                          (This file)
├── 📋 requirements.txt                   (Dependencies)
│
├── 📊 Data/
│   └── creditcard.csv                    (Dataset - auto-downloaded)
│
├── 📈 Visualizations/
│   ├── class_distribution.png
│   ├── amount_distribution.png
│   ├── time_distribution.png
│   ├── box_plot_amount.png
│   └── fraud_scatter_plot.png
│
└── 🤖 Models/
    ├── fraud_detection_model.h5          (Trained model)
    └── model_weights.h5                  (Weights only)
```

---

## 💻 Code Snippets

### Loading Data
```python
import kagglehub

df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    handle="mlg-ulb/creditcardfraud",
    path="creditcard.csv"
)
```

### Feature Scaling
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)
```

### Building Model
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

model = Sequential([
    Dense(128, activation='relu', input_dim=30),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

## 📊 Visualizations Included

<table>
<tr>
<td align="center">

**Count Plot**
Fraudulent vs Legitimate

</td>
<td align="center">

**Amount Distribution**
Transaction amounts histogram

</td>
<td align="center">

**Time Distribution**
Transaction timing pattern

</td>
</tr>
<tr>
<td align="center">

**Box Plot**
Amount outliers by class

</td>
<td align="center">

**Scatter Plot**
Fraudulent transactions timeline

</td>
<td align="center">

**Correlation Heatmap**
Feature relationships

</td>
</tr>
</table>

---

## ⚠️ Important Notes & Considerations

### 🔒 Privacy & Security
- ✅ V1-V28 are PCA-transformed for privacy
- ✅ No personally identifiable information
- ✅ Safe for public research

### 📌 Data Characteristics
- Features are already preprocessed and normalized
- Time span covers approximately 2 days
- Amount varies significantly by fraud type
- No temporal dependencies in sequence

### 🎯 Best Practices
1. **Always** use stratified train-test split
2. **Never** scale before train-test split
3. **Use** appropriate metrics (not just accuracy)
4. **Monitor** for data drift in production
5. **Validate** against business constraints

### 🚨 Production Considerations
- Implement model monitoring dashboard
- Set up fraud alert thresholds
- Create feedback loop for model retraining
- Establish A/B testing framework
- Document model decisions for compliance

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🐛 Report Issues
- Describe the problem clearly
- Include steps to reproduce
- Share relevant code/output

### 💡 Suggest Improvements
- New visualization ideas
- Better preprocessing techniques
- Alternative model architectures
- Performance optimizations

### 🔄 Submit Pull Requests
```bash
git checkout -b feature/your-feature-name
git commit -am 'Add your changes'
git push origin feature/your-feature-name
```

---

## 📚 Learning Resources

### 📖 Recommended Reading
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Handling Imbalanced Data](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-machine-learning/)

### 🎓 Related Concepts
- Imbalanced classification
- Deep learning fundamentals
- Model evaluation metrics
- Data preprocessing techniques

### 📺 Video Tutorials
- [Neural Networks with TensorFlow](https://www.youtube.com/results?search_query=tensorflow+neural+networks)
- [Fraud Detection Overview](https://www.youtube.com/results?search_query=fraud+detection+machine+learning)

---

## 📝 License

<div align="center">

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**Free to use, modify, and distribute** ✅

</div>

---

## 👤 Author & Contact

<div align="center">

**MrMaher19**

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=for-the-badge&logo=github)](https://github.com/MrMaher19)

</div>

---

## 🙏 Acknowledgments

<div align="center">

- **MLG-ULB** for providing the dataset
- **Kaggle** for dataset hosting and API
- **TensorFlow/Keras** for deep learning framework
- **Open Source Community** for tools and libraries

</div>

---

## ⭐ Show Your Support

If this project helped you, please consider:
- ⭐ Starring the repository
- 🔗 Sharing with others
- 📝 Providing feedback
- 🤝 Contributing improvements

---

<div align="center">

**Made with ❤️ for fraud detection and machine learning enthusiasts**

*Last Updated: July 2, 2026*

</div>
