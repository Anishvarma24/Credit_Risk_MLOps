# Credit Risk Prediction System

## Project Overview

The Credit Risk Prediction System is a Machine Learning and MLOps-based application that predicts the credit risk of a loan applicant using financial and historical information.

The system provides:

- Machine Learning based credit risk prediction
- Experiment tracking using MLflow
- Dataset versioning using DVC
- Remote storage using DagsHub
- REST API using FastAPI
- Containerization using Docker
- Data drift monitoring using Evidently AI
- Controlled model retraining
- CI automation using GitHub Actions
- Web-based frontend for prediction

---

## Team Members

- ALAPATI AASRITH
- KUTCHARLAPATI ANISH VARMA
- MADASU SAI PAVAN

---

## Project Objective

The main objective is to develop an automated credit-risk assessment system that can consistently evaluate an applicant's financial and historical data and classify the applicant as:

- Good Risk
- Bad Risk

The project also demonstrates an end-to-end MLOps workflow for managing the machine learning lifecycle.

---

## Dataset

The project uses the UCI Statlog German Credit Data dataset.

Dataset details:

- Number of records: 1000
- Number of columns: 25
- Features: 24
- Target: 1

Target mapping used in the project:

- 1 → Good Risk (0)
- 2 → Bad Risk (1)

Target distribution:

- Good Risk: 700
- Bad Risk: 300

---

## Machine Learning Model

The project uses:

**Logistic Regression**

Training process:

1. Load the dataset
2. Separate features and target
3. Convert target values
4. Split data into training and testing sets
5. Train Logistic Regression
6. Evaluate the model
7. Save the trained model

Train/Test split:

- Training data: 80%
- Testing data: 20%
- Random State: 42

---

## Model Performance

Model:

**Logistic Regression**

Accuracy:

**77%**

Classification performance:

| Class | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Good Risk | 0.80 | 0.90 | 0.85 |
| Bad Risk | 0.67 | 0.47 | 0.55 |

Overall Accuracy: **0.77**

---

## MLOps Architecture

```text
                    Dataset
                       |
                       v
                 Data Versioning
                     DVC
                       |
                       v
                    DagsHub
                       |
                       v
                Data Processing
                       |
                       v
                Model Training
                       |
                       v
                 ML Evaluation
                       |
                       v
                    MLflow
                       |
                       v
              Trained ML Model
                       |
              +--------+--------+
              |                 |
              v                 v
          FastAPI            Monitoring
              |              Evidently
              v                 |
           Docker               |
              |                 v
              +----------> Retraining
                       |
                       v
                 GitHub Actions