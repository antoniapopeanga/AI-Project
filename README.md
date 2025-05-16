# Artificial Intelligence Class
**L2 International - University of Bordeaux**  
*Supervised Learning: Decision Trees & Neural Networks*

> This repository contains both the **lab work completed throughout the semester** and the **final two projects**:  
> a Decision Tree for diabetes prediction and a Neural Network for digit classification using the MNIST dataset.

---

## 📦 Combined Imports for Both Projects

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
```

---

## 🌐 Dataset Overview

### 1. Diabetes Prediction Dataset

```python
diabetes_data = pd.read_csv("pima-indians-diabetes.csv")
print(diabetes_data.info())
```

**Characteristics:**
- 768 samples with 9 features  
- Binary classification (diabetic/non-diabetic)  
- Key features: `Glucose`, `BMI`, `Insulin`, `Age`

---

### 2. MNIST Handwritten Digits

```python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print(f"Training images shape: {train_images.shape}")
```

**Characteristics:**
- 60,000 training + 10,000 test samples  
- 28×28 grayscale images  
- 10 classes (digits 0–9)

---

## 🌲 Decision Tree Implementation

### 🔧 Data Preparation

```python
X = diabetes_data[["pregnancies", "insulin", "bmi", "glucose", "bp", "diabetespedigree"]]
y = diabetes_data["diabetic"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 📚 Model Training & Evaluation

```python
clf = DecisionTreeClassifier(max_depth=3, min_samples_split=10)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

**Results:**

| Metric           | Value     |
|------------------|-----------|
| Accuracy         | 77.49%    |
| True Positives   | 39        |
| False Positives  | 18        |

### 📊 Visualization

```python
plt.figure(figsize=(12,8))
plot_tree(clf, feature_names=X.columns, class_names=["Healthy", "Diabetic"], 
          filled=True, rounded=True)
plt.show()
```

---

## 🧠 Neural Network Implementation

### 🔧 Data Preprocessing

```python
# MNIST Flattening and Normalization
train_images = train_images.reshape(-1, 784).astype("float32") / 255
test_images = test_images.reshape(-1, 784).astype("float32") / 255

# Label Encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```

### 🏗️ Model Architecture

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### 🏃 Training Process

```python
history = model.fit(train_images, train_labels,
                    epochs=15,
                    batch_size=128,
                    validation_split=0.2)
```

**Performance Metrics:**

| Dataset     | Accuracy | Loss   |
|-------------|----------|--------|
| Training    | 99.2%    | 0.025  |
| Validation  | 98.1%    | 0.078  |
| Test        | 98.0%    | 0.082  |

---

## 🔍 Comparative Analysis

| Aspect            | Decision Trees                        | Neural Networks                   |
|-------------------|----------------------------------------|-----------------------------------|
| ✅ Interpretability | Easy to visualize & understand         | ❌ Black-box model                |
| ⚡ Speed           | Fast to train                          | Requires more training time       |
| 📈 Accuracy        | Moderate (77.49%)                      | High (98.0%)                      |
| 💡 Use Case        | Quick, interpretable models            | Complex pattern recognition       |
| ⚙️ Resource Usage  | Low                                    | Higher computational cost         |

---
