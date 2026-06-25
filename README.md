# Product Attribute Extraction using NLP + FastAPI

## 📌 Project Overview

This project is an AI/NLP based system that converts unstructured fashion product descriptions into structured product attributes.

The system takes a product description as input and extracts important attributes:

* Silhouette
* Fabric
* Neckline
* Sleeve
* Length
* Embellishment
* Color
* Category

The project uses Machine Learning techniques with a TF-IDF based text processing pipeline and Logistic Regression models. The trained models are deployed using FastAPI.

---

## 🚀 Features

✅ NLP based text processing
✅ Multi-attribute extraction
✅ Machine Learning model training
✅ Saved trained models
✅ FastAPI REST API
✅ JSON response format

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* FastAPI
* Joblib

---

## 📂 Project Structure

```
Product-Attribute-Extraction-NLP/

│
├── app/
│   └── main.py
│       FastAPI application

│
├── data/
│   └── data.csv
│       Labeled product dataset

│
├── models/
│   ├── train.py
│   ├── fabric.pkl
│   ├── neck.pkl
│   ├── sleeve.pkl
│   ├── color.pkl
│   ├── category.pkl
│   ├── silhouette.pkl
│   ├── length.pkl
│   ├── embellishment.pkl
│   └── vectorizer.pkl

│
└── README.md
```

---

# 🧠 Approach

## 1. Dataset Preparation

A labeled dataset is created containing:

* Product description
* Corresponding attribute labels

Example:

Input:

```
Floor length chiffon dress with V neckline in sage green
```

Output:

```
Fabric: Chiffon
Neckline: V Neck
Color: Green
Length: Full Length
```

---

## 2. Text Processing

Product descriptions are converted into numerical features using:

```
TF-IDF Vectorization
```

The vectorizer uses:

* Unigrams
* Bigrams

to capture important words and combinations.

---

## 3. Model Training

Separate Logistic Regression models are trained for each attribute:

* Fabric Model
* Neckline Model
* Sleeve Model
* Color Model
* Category Model
* Silhouette Model
* Length Model
* Embellishment Model

---

## 4. Model Saving

Trained models are saved using Joblib:

```
fabric.pkl
color.pkl
category.pkl
vectorizer.pkl
```

These models are loaded during API execution.

---

# 📊 Evaluation Metrics

Models are evaluated using:

* Accuracy Score
* Weighted F1 Score

Example evaluation:

```
Category Accuracy: 0.90+

Silhouette Accuracy: 0.72+

Length Accuracy: 0.81+
```

---

# 🌐 API Implementation

## Run Application

Install dependencies:

```
pip install -r requirements.txt
```

Start FastAPI server:

```
uvicorn app.main:app --reload
```

Server:

```
http://127.0.0.1:8000
```

---

# API Endpoint

## POST /extract

### Request

```json
{
 "product_description":
 "Floor length chiffon bridesmaid dress with V neckline in sage green"
}
```

---

### Response

```json
{
 "input":
 "Floor length chiffon bridesmaid dress with V neckline in sage green",

 "extracted_attributes":
 {
  "Silhouette":"Gown",
  "Fabric":"Chiffon",
  "Neckline":"V Neck",
  "Sleeve":"Sleeveless",
  "Length":"Full Length",
  "Embellishment":"None",
  "Color":"Green",
  "Category":"Wedding Dress"
 }
}
```

---




# 👨‍💻 Author

Rohit 