# 💧 Groundwater Quality Monitoring System

An AI-powered groundwater quality assessment and decision-support system using Machine Learning and environmental standards (BIS & WHO).  
This project analyzes groundwater samples, detects anomalies, computes Water Quality Index (WQI), and checks drinking water compliance.

---

## 📌 Project Overview

Groundwater is a major source of drinking and irrigation water. However, contamination due to industrialization, agriculture, and natural geochemical processes poses serious health risks.

This project provides an intelligent system that:
- Evaluates groundwater quality using physicochemical parameters
- Detects abnormal or unsafe samples using Machine Learning
- Computes Water Quality Index (WQI)
- Validates results against BIS and WHO standards
- Assists decision-making for water safety

The system is suitable for **academic projects, environmental monitoring, and ML portfolios**.

---

## 🎯 Objectives

- To analyze groundwater quality using scientific parameters  
- To detect anomalous water samples using ML models  
- To compute Water Quality Index (WQI)  
- To check drinking water compliance using BIS & WHO standards  
- To provide clear, interpretable results for decision support  

---

## 🧠 Machine Learning Models Used

### 1️⃣ Autoencoder (PyTorch)
- Used for anomaly detection  
- Identifies abnormal groundwater samples that deviate from normal patterns  

### 2️⃣ PCA (Principal Component Analysis)
- Used for dimensionality reduction  
- Improves model efficiency and visualization  

### 3️⃣ Random Forest Cluster Emulator
- Assigns groundwater samples to learned quality clusters  

---

## 📊 Dataset Description

The dataset contains groundwater physicochemical parameters such as:
- pH  
- Total Dissolved Solids (TDS)  
- Nitrate  
- Chloride  
- Sulfate  
- Fluoride  
- Iron  
- Total Hardness  
- Electrical Conductivity  

Data preprocessing includes:
- Missing value handling (median-based)  
- Feature scaling using StandardScaler  
- Dimensionality reduction using PCA  

---

## ⚙️ System Architecture

1. Input groundwater sample parameters  
2. Data preprocessing & scaling  
3. PCA transformation  
4. ML-based anomaly detection  
5. WQI computation  
6. BIS & WHO compliance checking  
7. Final quality classification & insights  

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
pip install -r requirements.txt

### 2️⃣ Run the Application
python app.py

## 📈 Results

Accurately detects anomalous groundwater samples

Computes WQI score and quality category

Flags unsafe parameters exceeding permissible limits

Helps identify water samples unsuitable for drinking

## 🔮 Future Scope

Integration with real-time IoT groundwater sensors

Expansion to large-scale regional groundwater monitoring

Advanced deep learning models for predictive analysis

Cloud deployment for real-time decision support

Mobile application for field-level usage

## 🛠 Technologies Used

Python

Machine Learning

PyTorch

Scikit-learn

Pandas & NumPy

Matplotlib

Jupyter Notebook

## 👨‍🎓 Author

Om Pandey
Computer Science Engineering
Aspiring Machine Learning Engineer

## 📜 Disclaimer

This system provides decision-support insights based on machine learning and illustrative data.
It should be used alongside laboratory testing for critical real-world applications.