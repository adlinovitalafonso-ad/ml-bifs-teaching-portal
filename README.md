# Machine Learning for BIFS — Teaching Portal

An interactive Streamlit application designed for teaching Machine Learning concepts in the BIFS programme.

## Overview

The portal provides classroom-oriented demonstrations, hands-on experimentation, and interpretation support across major Machine Learning topics.

It is intended primarily for educational use and focuses on helping students understand how model choices, parameters, data characteristics, and evaluation metrics affect results.

## Learning Areas

The portal currently includes:

- Regression
  - Simple Linear Regression
  - Multiple Linear Regression
  - Polynomial Regression
  - Ridge Regression
  - Lasso Regression

- Classification
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Support Vector Machine

- Unsupervised Learning
  - K-Means Clustering
  - Hierarchical Clustering
  - Principal Component Analysis

- Semi-Supervised Learning
  - Wrapper / Self-Training
  - Clustering-Based Semi-Supervised Learning

- Reinforcement Learning
  - Multi-Armed Bandit demonstration

- Natural Language Processing
  - VADER Sentiment Analysis
  - FinBERT
  - TF-IDF keyword analysis

## Teaching Features

The application includes:

- Built-in teaching datasets and CSV upload
- Interactive model controls
- Automatic handling of categorical predictors
- Class imbalance options
- Confusion matrix and classification metrics
- Feature importance
- PCA eigenvalues, component weights, loadings, and scores
- Before-and-after visualizations for semi-supervised learning
- Interactive unseen-case prediction
- Downloadable model results
- Result-specific interpretation and recommendations
- Student access control
- Session and usage analytics
- Feedback collection

## Running the App Locally

### 1. Install Python

Python 3.10 or later is recommended.

### 2. Install dependencies

From the project folder, run:

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

## Project Structure

```text
ml-bifs-teaching-portal/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

Local configuration and analytics files should not be committed to GitHub.

Recommended `.gitignore` entries:

```text
.streamlit/secrets.toml
ml_bifs_analytics.db
__pycache__/
*.pyc
```

## Streamlit Secrets

Access codes and deployment-specific settings should be stored in Streamlit Secrets rather than directly in the public source code.

Typical settings include:

```toml
ML_PORTAL_ACCESS_CODE = "your-student-access-code"
ML_PORTAL_ADMIN_CODE = "your-admin-code"
ML_PORTAL_ACTIVE_FROM = "YYYY-MM-DDTHH:MM"
ML_PORTAL_ACTIVE_UNTIL = "YYYY-MM-DDTHH:MM"
ML_PORTAL_MAX_USERS = 30
ML_PORTAL_SESSION_TIMEOUT_MINUTES = 10
```

Do not commit `secrets.toml` to a public repository.

## Intended Use

This application is designed for classroom demonstrations, student experimentation, and educational interpretation of Machine Learning methods.

It is not intended for production decision-making or high-stakes financial, medical, legal, or operational use.

## Attribution

**Machine Learning for BIFS — Teaching Portal**

Concept and teaching design: **Prof. Adlino Afonso**  
Developed with AI assistance  
For educational use only
