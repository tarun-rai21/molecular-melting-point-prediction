# 🧪 Melting Point Prediction – Kaggle Competition

## 📌 Competition  
Thermophysical Property: Melting Point Prediction Challenge on Kaggle

---

## 🎯 Objective  
Predict the **melting point (Tm)** of organic molecules using machine learning.  
Melting point is a key thermophysical property important in:

- Drug design  
- Material science  
- Chemical engineering  

---

## 📊 Dataset Overview

| Item | Details |
|------|--------|
| Total molecules | 3328 |
| Training set | 2662 molecules |
| Test set | 666 molecules |
| Target variable | `Tm` (Kelvin) |
| Evaluation metric | **Mean Absolute Error (MAE)** |

Each molecule is represented using:

- **SMILES string** (molecular structure)
- Group contribution descriptors

---

## ⚙️ Machine Learning Pipeline

This project follows a **modular cheminformatics ML workflow**:

1. Data preprocessing  
2. Molecular feature extraction  
3. Model training & validation  
4. Final prediction on unseen molecules  
5. Submission file generation  

---

## 🔬 Feature Engineering Approaches (Experimentation)

This project explored multiple molecular representations:

| Solution | Features Used | Purpose |
|----------|---------------|--------|
| **Solution 1** | Provided numeric group descriptors only | Baseline model |
| **Solution 2** | RDKit molecular descriptors from SMILES | Capture physicochemical properties |
| **Solution 3** | RDKit descriptors + numeric features | Combine structural + provided info |
| **Solution 4 (Final)** | **RDKit descriptors + Morgan fingerprints** | Add molecular topology & structure information |

### Final Feature Set Includes:
- Molecular weight  
- TPSA  
- LogP  
- Hydrogen bond donors/acceptors  
- Ring counts  
- Surface area descriptors  
- **2048-bit Morgan fingerprints (radius = 2)**  

This combination provided the best predictive performance.

---

## 🤖 Model

- **XGBoost Regressor**
- Log-transformed target (`log1p(Tm)`)
- Train–validation split
- Metric: MAE

---

## 🏗 Project Structure

```bash
molecular-melting-point-prediction/
│
├── notebooks/
│   └── solution_4_rdkit_morgan.ipynb   # Final best-performing notebook
│
├── src/
│   ├── preprocess.py   # Data cleaning + SMILES validation
│   ├── features.py     # RDKit descriptors + Morgan fingerprints
│   ├── model.py        # XGBoost regression (log-transformed target)
│   └── predict.py      # Submission file generation
│
├── submissions/        # Generated Kaggle submissions
│
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── .gitignore          # Ignored files
```

--- 

## 🧠 Skills Demonstrated 
- Cheminformatics feature engineering
- RDKit molecular descriptor extraction
- Morgan fingerprint generation
- Regression modeling
- Feature experimentation
- ML pipeline modularization
- Kaggle competition workflow

--- 
## 🚀 Technologies Used 
- Python
- RDKit
- Pandas
- Scikit-learn
- XGBoost

--- 

This project demonstrates applying machine learning to **scientific molecular data**, combining domain-specific feature engineering with modern regression techniques.
