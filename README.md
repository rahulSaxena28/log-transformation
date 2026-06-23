# 📊 Log Transformation on Titanic Dataset

A hands-on exploration of **Log Transformation** as a feature engineering technique, applied on the Titanic dataset using two machine learning classifiers — **Logistic Regression** and **Decision Tree**.

---

## 🎯 Objective

To understand when and why to apply log transformation by:
- Visually analyzing data distribution using **Histogram (PDF)** and **Q-Q Plots**
- Comparing model accuracy **before** and **after** transformation
- Applying transformation selectively using `ColumnTransformer`

---

## 📂 Dataset

**Titanic Dataset** — using only 3 columns:

| Column     | Type        | Role    |
|------------|-------------|---------|
| 🎯 `Survived` | Target (0/1)| Label   |
| 🧓 `Age`      | Numerical   | Feature |
| 💰 `Fare`     | Numerical   | Feature |

> Missing values in `Age` are filled with the **column mean** before processing.

---

## 🔁 Workflow

```
Load Data → Handle Nulls → Train-Test Split
    ↓
Visualize (Histogram + Q-Q Plot) → Baseline Model Accuracy
    ↓
Apply Log Transformation (np.log1p)
    ↓
Visualize Again → Compare Model Accuracy
    ↓
Selective Transformation (only Fare) → Final Comparison
```

---

## 📦 Libraries Used

| Library | Purpose |
|---------|---------|
| 🐼 `pandas` | Data loading & manipulation |
| 🔢 `numpy` | Numerical operations & log1p |
| 🎨 `seaborn` | Histogram / KDE plots |
| 📉 `matplotlib` | Plot rendering |
| 📐 `scipy.stats` | Q-Q Plot |
| 🤖 `scikit-learn` | ML models & transformers |

---

## 🔬 What is Log Transformation?

Log Transformation is a **mathematical technique** that applies `log(x)` to feature values to:

- Reduce the effect of **right-skewed** distributions
- Bring data closer to a **normal (Gaussian) distribution**
- Reduce the influence of **extreme outliers**

### Formula used:
```python
np.log1p(x)  # same as log(1 + x), safe for zero values
```

> ✅ **Why `log1p` instead of `log`?**
> `np.log(0)` returns `-inf`, which breaks models.
> `np.log1p(0) = log(1+0) = 0` — safe for zero values!

---

## ⚠️ Important Notes on Log Transformation

### ✅ When TO apply it:
- 📐 Feature has a **right-skewed (positive skew)** distribution
- 📉 Q-Q Plot shows data **curving away** from the diagonal line
- 💥 Feature contains **large outlier values** (like `Fare` in Titanic)
- 🤖 You are using **distance-based or linear models** (e.g., Logistic Regression, SVM, KNN) that assume normality

### ❌ When NOT to apply it:
- 🔔 Feature already has a **normal or uniform distribution** (like `Age` here — log transformation barely helps it)
- ➖ Feature contains **negative values** — log of negative numbers is undefined
- 🌳 You are using **tree-based models** (e.g., Decision Tree, Random Forest) — they split on thresholds and are **not affected by scale or distribution**

### 🔍 How to decide — use these two tools:
| Tool | What it shows |
|------|--------------|
| **Histogram / KDE Plot** | Overall shape of distribution — skewness, peaks, outliers |
| **Q-Q Plot** | How closely data follows a normal distribution — points should lie on the diagonal |

---

## 🧪 Experiment Results

### Before Transformation:
| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | ~XX%     |
| Decision Tree       | ~XX%     |

### After Log Transformation (both features):
| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | ~XX%     |
| Decision Tree       | ~XX%     |

### After Selective Transformation (only `Fare`):
| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | ~XX%     |
| Decision Tree       | ~XX%     |

> 💡 Fill in the actual accuracy values from your notebook output!

---

## 💡 Key Takeaway

> Not all features need transformation. Always **visualize first**, then decide.
> Log transformation is a tool — use it **only where the data calls for it**.

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/log-transformation-titanic.git
cd log-transformation-titanic
pip install pandas numpy seaborn matplotlib scipy scikit-learn
jupyter notebook Log_transformer.ipynb
```

---

## 📁 Project Structure

```
log-transformation-titanic/
│
├── Log_transformer.ipynb     # Main notebook
├── Titanic-Dataset.csv       # Dataset
└── README.md                 # You are here
```

---

## 🧠 Concepts Covered

- 🛠️ Feature Engineering
- 📈 Log Transformation (`np.log1p`)
- ⚙️ `FunctionTransformer` from sklearn
- 🎛️ `ColumnTransformer` for selective feature transformation
- 🔍 Distribution Analysis: Histogram, KDE, Q-Q Plot
- ⚖️ Model comparison: Before vs After preprocessing

---

## 👤 Author

Made with 🔥 while learning Machine Learning preprocessing techniques.
