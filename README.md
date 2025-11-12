# 🕵️‍♂️ Credit Card Fraud Detection with PyTorch

A machine learning project designed to detect fraudulent credit card transactions using a neural network.  
This project explores and models the **Credit Card Fraud Dataset** from Kaggle, performing exploratory data analysis (EDA) and training a deep learning classifier using **PyTorch**.

---

## 📊 Project Overview

Credit card fraud detection is a classic example of an **imbalanced classification problem**, where fraudulent transactions represent only a small fraction of all transactions.

In this project, we:

- 🧹 Import and clean the dataset  
- 📈 Explore and visualize key features affecting fraud likelihood  
- ⚖️ Balance the data using undersampling  
- 🧠 Build and train a neural network to classify transactions as fraudulent or legitimate  
- ✅ Evaluate model performance through training loss and inference accuracy  

---

## 🧩 Key Features

- Exploratory Data Analysis (EDA) with **matplotlib** and **seaborn**  
- Data balancing using `imblearn.RandomUnderSampler`  
- Feature correlation analysis  
- Neural Network implementation with **PyTorch**  
- Train/Test split with stratification  
- Batch training for gradient stability  

---

## ⚙️ Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/bradyrichardson/fraud-detection.git
cd fraud-detection
```

### 2. Install Dependencies
```bash
pip install torch torchvision kagglehub pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### 3. Download the Dataset
You can use the **KaggleHub** library to fetch the dataset automatically:

```python
import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "card_transdata.csv"
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "dhanushnarayananr/credit-card-fraud",
    file_path
)
```

Alternatively, download it manually from Kaggle and place it in the project directory.

---

## 🧼 Data Cleaning

- Removed NaN values and duplicates (though dataset was already clean)  
- Verified integrity with `df.shape` before and after cleaning  

---

## 🔍 Exploratory Data Analysis

Analyzed relationships between features and fraudulent transactions:

| Feature | Insight |
|----------|----------|
| `distance_from_home` | 40% of frauds occur far from home (>75th percentile) |
| `distance_from_last_transaction` | 30% of frauds occur far from the previous transaction |
| `ratio_to_median_purchase_price` | 38% of frauds have unusually high purchase ratios |
| `online_order` | 95% of frauds were online transactions |
| `used_pin_number` | Less than 1% of frauds used PIN |
| `used_chip` | ~25% of frauds used a chip (possible data inconsistency) |

### Visualization Examples
- 📦 Boxplots for distance and ratio features  
- 🌐 Scatterplots for purchase ratios  
- 📊 Bar charts comparing online/offline and repeat/new retailer transactions  
- 🔥 Correlation heatmap across all features  

---

## 🧠 Model Architecture

The neural network is implemented as follows:

```python
import torch.nn as nn

class FraudDetectionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 14),
            nn.ReLU(),
            nn.Linear(14, 28),
            nn.ReLU(),
            nn.Linear(28, 14),
            nn.ReLU(),
            nn.Linear(14, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
```

### Training Details
| Parameter | Value |
|------------|--------|
| Loss Function | CrossEntropyLoss |
| Optimizer | SGD (momentum=0.9) |
| Learning Rate | 0.0001 |
| Epochs | 100 |
| Batch Size | 64 |

### Data Handling
- Used **stratified train/test split** to preserve class ratio  
- Applied **undersampling** (50/50 split) on the training set to handle imbalance  

---

## 📈 Results

| Metric | Description |
|---------|-------------|
| Fraud Rate | 8.74% of all transactions |
| Training Loss | Decreased steadily over 100 epochs (from 0.59 → ~0.20) |
| Best Predictive Features | `distance_from_home`, `ratio_to_median_purchase_price`, `online_order` |

Visualization of training loss over epochs shows **consistent convergence**, suggesting good model learning behavior.

---

## 🧮 Future Improvements

- Implement **SMOTE** or **class-weighted loss** for more balanced learning  
- Add **feature scaling/normalization**  
- Introduce **dropout** and **batch normalization** for regularization  
- Evaluate with **precision, recall, F1-score, and ROC curves**  
- Experiment with **tree-based models** (e.g., XGBoost, Random Forest) for comparison  

---

## 🧰 Tech Stack

- 🐍 Python 3.12  
- 🔥 PyTorch  
- 🧮 NumPy / Pandas  
- 📊 Matplotlib / Seaborn  
- 🧠 scikit-learn  
- ⚖️ imbalanced-learn  
- 📦 KaggleHub  

---

## 📂 File Structure
```
fraud-detection/
│
├── fraud_detection.ipynb       # Main notebook (EDA + model training)
├── README.md                   # Project documentation
├── card_transdata.csv          # Dataset (if downloaded manually)
└── requirements.txt            # List of dependencies
```

---

## 👤 Author

**Brady Richardson**  
📧 [bradyrr33@gmail.com(mailto:bradyrr33@gmail.com)  
💼 [[LinkedIn](#)](https://www.linkedin.com/in/brady-r-richardson/)

---

⭐ *Built with PyTorch — Deep Learning for Fraud Detection Made Simple.* 🧠
