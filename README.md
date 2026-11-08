# Heart Disease Classification — ML Assignment 2

**BITS Pilani | M.Tech (AIML) | Machine Learning | Assignment 2**

---

## a. Problem Statement

Predict the presence or absence of heart disease in a patient based on clinical and diagnostic attributes. This is a **binary classification** problem where the target variable indicates whether a patient has heart disease (1) or not (0). Early and accurate detection of heart disease is critical for timely medical intervention, and machine learning models can assist clinicians in making faster diagnostic decisions.

---

## b. Dataset Description

- **Dataset:** Heart Disease Dataset (based on UCI Heart Disease repository)
- **Source:** UCI Machine Learning Repository — Cleveland, Hungary, Switzerland, VA Long Beach
- **Instances:** 920
- **Features:** 13 (all numeric)
- **Target:** Binary — `0` (No Heart Disease), `1` (Heart Disease Present)
- **Target Distribution:** Approximately balanced (463 vs 457)

### Feature Details

| # | Feature     | Description                                      | Type       |
|---|-------------|--------------------------------------------------|------------|
| 1 | age         | Age of the patient in years                      | Numeric    |
| 2 | sex         | Sex (1 = male, 0 = female)                       | Binary     |
| 3 | cp          | Chest pain type (0–3)                            | Categorical|
| 4 | trestbps    | Resting blood pressure (mm Hg)                   | Numeric    |
| 5 | chol        | Serum cholesterol (mg/dl)                        | Numeric    |
| 6 | fbs         | Fasting blood sugar > 120 mg/dl (1 = true)       | Binary     |
| 7 | restecg     | Resting ECG results (0–2)                        | Categorical|
| 8 | thalach     | Maximum heart rate achieved                      | Numeric    |
| 9 | exang       | Exercise-induced angina (1 = yes)                | Binary     |
|10 | oldpeak     | ST depression induced by exercise                | Numeric    |
|11 | slope       | Slope of the peak exercise ST segment (0–2)      | Categorical|
|12 | ca          | Number of major vessels colored by fluoroscopy (0–3)| Numeric  |
|13 | thal        | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect) | Categorical |

---

## c. GitHub Repository Link

**Repository:** [[https://github.com/YOUR_USERNAME/heart-disease-ml-assignment](https://github.com/Tanooj-P/2026aj05613_heart-disease-ml-assignment2.git)]([https://github.com/YOUR_USERNAME/heart-disease-ml-assignment](https://github.com/Tanooj-P/2026aj05613_heart-disease-ml-assignment2.git))

---

## d. Models Used

Five classification models were trained on an 80-20 train-test split (stratified) with StandardScaler preprocessing.

### Comparison Table — Evaluation Metrics

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.7989   | 0.8886 | 0.7935    | 0.8022 | 0.7978 | 0.5979 |
| Decision Tree          | 0.7065   | 0.7222 | 0.7403    | 0.6264 | 0.6786 | 0.4169 |
| KNN                    | 0.7717   | 0.8082 | 0.8182    | 0.6923 | 0.7500 | 0.5491 |
| Naive Bayes            | 0.7772   | 0.8721 | 0.7976    | 0.7363 | 0.7657 | 0.5555 |
| Support Vector Machine | 0.7717   | 0.8584 | 0.7634    | 0.7802 | 0.7717 | 0.5437 |

### Observations on Model Performance

| ML Model Name          | Observation about model performance |
|------------------------|-------------------------------------|
| Logistic Regression    | Best overall performer with the highest accuracy (0.7989), F1 score (0.7978), and AUC (0.8886). Despite being a linear model, it captures the underlying patterns effectively, suggesting the decision boundary is approximately linear. Balanced precision and recall indicate reliable predictions for both classes. |
| Decision Tree          | Weakest performer across all metrics. Accuracy of 0.7065 and lowest AUC (0.7222) indicate poor generalization. The relatively low recall (0.6264) means it misses many positive cases. Prone to overfitting despite max_depth constraint. Not recommended for this dataset. |
| KNN                    | Achieves the highest precision (0.8182) among all models, meaning fewer false positives. However, recall is moderate (0.6923), so it misses some disease cases. Good choice when false positives are costly. Instance-based learning captures local patterns but is sensitive to feature scaling. |
| Naive Bayes            | Second-best AUC score (0.8721), showing strong probabilistic calibration. Reasonable balance across all metrics. The conditional independence assumption works reasonably well here despite likely feature correlations. Computationally efficient and handles small datasets well. |
| Support Vector Machine | Balanced precision (0.7634) and recall (0.7802), with good AUC (0.8584). The RBF kernel captures non-linear relationships. Performance is competitive but not the best, possibly due to limited hyperparameter tuning. |
| **Overall Winner**     | **Logistic Regression** — achieves the best accuracy, AUC, F1 score, and MCC. Its strong and balanced performance across all metrics makes it the most reliable model for this heart disease classification task. |

---

## e. Streamlit App Features

- **CSV Upload:** Upload test data in CSV format with all 13 features and target column
- **Model Selection:** Dropdown to select any of the 5 trained models
- **Metrics Display:** All 6 evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- **Confusion Matrix:** Visual heatmap for the selected model
- **ROC Curve:** ROC curve with AUC value
- **Classification Report:** Detailed precision, recall, f1-score per class
- **Model Comparison:** Side-by-side comparison table of all 5 models with bar chart

---

## g. Project Structure

```
2026aj05613_heart-disease-ml-assignment2/
├── app.py                  # Streamlit web application
├── train_models.py         # Model training script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── test_data.csv           # Test dataset (184 samples)
├── heart_disease_full.csv  # Full dataset (920 samples)
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── support_vector_machine.pkl
    ├── scaler.pkl
    ├── feature_names.pkl
    └── results.pkl
```

---

## h. Live Streamlit App

**Deployed App:** [https://2026aj05613heart-disease-ml-assignment2.streamlit.app/)


---

## Technologies Used

- Python
- Streamlit
- scikit-learn
- pandas, NumPy
- matplotlib, seaborn
